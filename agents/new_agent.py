import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
import logging


from .agent import Agent

try:
    # 用于构造“轻量复制”的球对象：避免 deepcopy 拷贝巨大 history（这是主要性能瓶颈）
    from pooltool.objects.ball.datatypes import Ball, BallHistory
except Exception:  # pragma: no cover
    Ball = None
    BallHistory = None


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        在支持 SIGALRM 的平台（通常为 Unix/Linux）上，使用 signal.SIGALRM 实现硬超时。
        在 Windows 或不支持 SIGALRM/alarm 的环境、或非主线程中，自动降级为直接调用
        pt.simulate（不做硬超时），以避免因 SIGALRM 不存在导致评测直接失败。
    """
    # Windows 没有 SIGALRM；另外 signal.* 通常只能在主线程使用。
    if not (hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")):
        try:
            pt.simulate(shot, inplace=True)
            return True
        except Exception as e:
            logger.warning(f"物理模拟失败（无超时保护降级路径）: {e}")
            return False

    try:
        # 设置超时信号处理器（非主线程可能失败，此时降级）
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)  # 设置超时时间
        except Exception as e:
            logger.debug(f"无法启用SIGALRM超时保护，降级为直接模拟: {e}")
            pt.simulate(shot, inplace=True)
            return True

        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        try:
            signal.alarm(0)  # 取消超时
        except Exception:
            pass
        logger.warning(f"物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception:
        try:
            signal.alarm(0)  # 取消超时
        except Exception:
            pass
        raise
    finally:
        # 恢复原处理器（若 signal 在非主线程不可用，这里也要容错）
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

# ============================================

class NewAgent(Agent):
    """方案B优化版v2：几何瞄准与增强优化 Agent
    
    核心策略：
    1. 几何计算：精确计算球到袋口的瞄准点
    2. 物理模拟：验证击球可行性，避免犯规
    3. 智能选择：优先选择成功率高的目标球
    4. 安全击球：严格检查首球犯规和黑8风险
    5. 多重验证：增强首球检测和白球落袋检测
    """
    
    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575  # 标准台球半径（米）
        
        # 优化参数
        self.SIMULATION_COUNT = 30  # 增加模拟次数
        self.SIMULATION_TIMEOUT = 2  # 单次模拟超时
        self.POWER_REDUCTION = 0.9  # 力度衰减系数，降低白球落袋概率
        
        # 噪声参数（与poolenv保持一致）
        self.noise_std = {
            'V0': 0.1,      # 速度标准差
            'phi': 0.1,     # 水平角度标准差（度）
            'theta': 0.1,   # 垂直角度标准差（度）
            'a': 0.003,     # 横向偏移标准差
            'b': 0.003      # 纵向偏移标准差
        }
        # 评估加速：两阶段评估
        # - SCREEN: 用更少的带噪声采样做快速筛选（用于搜索/微调）
        # - FINAL: 对最终候选用全量采样复核，保持最终精度
        self.ROBUSTNESS_SAMPLES_FINAL = 10
        self.ROBUSTNESS_SAMPLES_SCREEN = 4
        self.ROBUST_TOPK = 6  # 仅对 Top-K 候选做全量鲁棒性评估

        # 综合评分权重（保持原先“模拟优先”的倾向）
        self.GEO_WEIGHT = 0.3
        self.SIM_WEIGHT = 0.7

        # 性能：袋口位置缓存（同一 decision 内 table 对象是同一个，缓存可避免重复遍历）
        self._pockets_cache_table_id = None
        self._pockets_cache = None

        logger.info("NewAgent  已初始化")
    
    def _get_ball_position(self, ball):
        """获取球的2D位置"""
        return np.array([ball.state.rvw[0][0], ball.state.rvw[0][1]])
    
    def _get_pocket_positions(self, table):
        """从球桌对象获取真实袋口位置"""
        pockets = []
        for pocket in table.pockets.values():
            pos = pocket.center
            pockets.append(np.array([pos[0], pos[1]]))
        return pockets

    def _get_pocket_positions_cached(self, table):
        table_id = id(table)
        if self._pockets_cache_table_id != table_id or self._pockets_cache is None:
            self._pockets_cache_table_id = table_id
            self._pockets_cache = self._get_pocket_positions(table)
        return self._pockets_cache

    def _clone_balls_for_sim(self, balls):
        """为仿真构造轻量的 balls 副本。

        关键：PoolEnv 每次真实击球后，ball.history 会非常长；
        直接 deepcopy 会把整个 history 拷贝一份，导致 NewAgent 极慢。
        仿真只需要当前 state + params，因此这里构造空 history 的 Ball。
        """
        # 若导入失败则退化到 deepcopy（不会影响正确性，但会慢）
        if Ball is None or BallHistory is None:
            return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        sim_balls = {}
        empty_hist = BallHistory.factory
        for bid, ball in balls.items():
            # 只 deep copy state，params/ballset/orientation 共享引用即可（只读）
            sim_balls[bid] = Ball(
                id=ball.id,
                state=copy.deepcopy(ball.state),
                params=ball.params,
                ballset=ball.ballset,
                initial_orientation=ball.initial_orientation,
                history=empty_hist(),
                history_cts=empty_hist(),
            )
        return sim_balls
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _is_break_state(self, balls, my_targets):
        """判断是否为开球局面：15 颗球处于“完美三角阵”时触发开球策略。

        说明：环境不提供 hit_count，且仅靠 bbox/半径聚类会误判/漏判。
        这里改用“接触图”判定：若 15 颗球之间存在大量距离约等于 2R 的近邻对，
        说明球处于紧密三角架（racked triangle）。
        """
        try:
            if my_targets == ['8']:
                return False
            object_ids = [str(i) for i in range(1, 16)]
            if 'cue' not in balls or balls['cue'].state.s == 4:
                return False
            for bid in object_ids:
                if bid not in balls or balls[bid].state.s == 4:
                    return False

            obj_positions = np.array([self._get_ball_position(balls[bid]) for bid in object_ids], dtype=float)
            centroid = obj_positions.mean(axis=0)
            cue_pos = self._get_ball_position(balls['cue'])
            cue_to_centroid = float(np.linalg.norm(cue_pos - centroid))
            # 白球离球堆应当较远（避免把“局部拥挤”误当开球）
            if cue_to_centroid < 0.35:
                return False

            # 统计距离接近 2R 的近邻对数量。
            # 5 行三角阵（15球）理论上约有 30 对“紧邻接触”。考虑数值误差，允许一定容差。
            two_r = 2.0 * self.BALL_RADIUS
            tol = two_r * 0.10  # 10% 容差，兼容不同实现/浮点误差

            close_pairs = 0
            neighbor_counts = np.zeros(len(object_ids), dtype=int)
            for i in range(len(object_ids)):
                for j in range(i + 1, len(object_ids)):
                    d = float(np.linalg.norm(obj_positions[i] - obj_positions[j]))
                    if abs(d - two_r) <= tol:
                        close_pairs += 1
                        neighbor_counts[i] += 1
                        neighbor_counts[j] += 1

            # 结构性约束：接触对数量足够多 + 多数球有 >=2 个近邻（非散开态）
            if close_pairs < 22:
                return False
            if int(np.sum(neighbor_counts >= 2)) < 12:
                return False

            # 额外约束：整体紧凑（防止散开后偶然出现很多 2R 距离）
            bbox = obj_positions.max(axis=0) - obj_positions.min(axis=0)
            bbox_diag = float(np.linalg.norm(bbox))
            if bbox_diag > 0.40:
                return False

            return True
        except Exception:
            return False

    def _break_decision(self, balls, my_targets, table):
        """开球快速策略：少量候选 + 极少评估。

        目标：避免在开球局面跑完整的 Top-K * 鲁棒评估流程。
        关键：必须尽量保证首球合法（首碰必须落在 my_targets 内，且不能先碰 8）。
        """
        cue_pos = self._get_ball_position(balls['cue'])
        object_ids = [str(i) for i in range(1, 16)]
        obj_positions = np.array([self._get_ball_position(balls[bid]) for bid in object_ids])
        rack_centroid = obj_positions.mean(axis=0)

        # 关键改进：环境规则要求“首碰必须是己方球”。
        # 如果我方是 stripe，而三角阵尖球通常是 1（solid），直接打中尖球会犯规。
        # 因此优先选择“直线可视”的己方球作为首碰目标。
        best_phi = None
        best_ref = None
        best_delta = None

        visible_targets = []
        for tid in my_targets:
            if tid == '8':
                continue
            if tid not in balls or balls[tid].state.s == 4:
                continue
            tpos = self._get_ball_position(balls[tid])
            phi_to_t = self._calculate_shot_angle(cue_pos, tpos)
            first_contact = self._get_first_contact_ball(cue_pos, phi_to_t, balls)
            if first_contact == tid:
                dist = self._calculate_distance(cue_pos, tpos)
                visible_targets.append((dist, tid, phi_to_t))

        if visible_targets:
            visible_targets.sort(key=lambda x: x[0])
            _, best_tid, best_phi = visible_targets[0]
            best_ref = f"direct:{best_tid}"
        else:
            base_phi = self._calculate_shot_angle(cue_pos, rack_centroid)
            # 在 base_phi 附近搜索一个“首碰合法”的角度（几何预测，代价极低）
            for delta in [0, -1.5, 1.5, -3, 3, -5, 5, -8, 8, -12, 12, -18, 18, -25, 25, -35, 35]:
                phi = (base_phi + delta) % 360
                first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
                if first_contact is None:
                    continue
                if first_contact == '8':
                    continue
                if first_contact in my_targets:
                    best_phi = phi
                    best_delta = abs(delta)
                    best_ref = f"scan:{first_contact}"
                    break

            if best_phi is None:
                best_phi = base_phi
                best_ref = "fallback:centroid"

        # 少量力度候选（开球不需要精细调 a/b/theta）
        candidates = []
        for v0 in [6.0, 6.6, 7.2]:
            candidates.append({'V0': v0, 'phi': best_phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0})

        # 轻量评估：只做一次无噪声仿真（开球不需要噪声鲁棒评估）
        best_action = candidates[0]
        best_score = -1e9
        for action in candidates:
            fast_score = self._simulate_and_evaluate(action, balls, my_targets, table, add_noise=False)
            if fast_score <= -5000:
                continue

            if fast_score > best_score:
                best_score = fast_score
                best_action = action

        logger.info(
            f"[NewAgent] 开球策略: phi={best_action['phi']:.2f}, V0={best_action['V0']:.2f}" +
            (f", delta={best_delta}" if best_delta is not None else "") +
            (f", ref={best_ref}" if best_ref is not None else "")
        )
        return best_action

    def _estimate_opponent_threat(self, balls_after, table, my_targets):
        """估算对手回合的“最强简单进攻机会”（纯几何近似，低开销）。

        返回：float，越大表示对手越容易直接进球。
        仅在本方动作结束且球权将交给对手时使用，用于安全性惩罚。
        """
        try:
            if 'cue' not in balls_after or balls_after['cue'].state.s == 4:
                return 0.0
            cue_pos = self._get_ball_position(balls_after['cue'])
            pockets = self._get_pocket_positions_cached(table)

            # 对手目标：当前桌面上“非我方目标球、非白球、非黑8”的所有活球
            opp_ids = []
            for bid, b in balls_after.items():
                if bid in ['cue', '8']:
                    continue
                if b.state.s == 4:
                    continue
                if bid not in my_targets:
                    opp_ids.append(bid)
            if not opp_ids:
                return 0.0

            best = -1e9
            for tid in opp_ids:
                tpos = self._get_ball_position(balls_after[tid])
                for pocket_pos in pockets:
                    aim_point = self._calculate_aim_point(tpos, pocket_pos)
                    if aim_point is None:
                        continue
                    # 路径是否清晰（只做直线阻挡检查）
                    if not self._check_path_clear(cue_pos, aim_point, balls_after, exclude_ids=['cue', tid]):
                        continue
                    if not self._check_path_clear(tpos, pocket_pos, balls_after, exclude_ids=['cue', tid]):
                        continue

                    # 难度估计：距离越短越好，切角越小越好
                    cue_to_aim = self._calculate_distance(cue_pos, aim_point)
                    target_to_pocket = self._calculate_distance(tpos, pocket_pos)

                    vec1 = np.array(aim_point) - np.array(cue_pos)
                    vec2 = np.array(pocket_pos) - np.array(tpos)
                    angle_bonus = 0.0
                    if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                        v1 = vec1 / np.linalg.norm(vec1)
                        v2 = vec2 / np.linalg.norm(vec2)
                        cos_angle = float(np.dot(v1, v2))
                        angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
                        # 直球/小切角：奖励更高
                        angle_bonus = max(0.0, 80.0 - angle)  # angle=0 -> 80

                    # 距离惩罚
                    dist_pen = 20.0 * (cue_to_aim + target_to_pocket)
                    score = angle_bonus - dist_pen
                    if score > best:
                        best = score

            return float(best) if best > -1e8 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_aim_point(self, target_pos, pocket_pos):
        """计算瞄准点（ghost ball position）"""
        direction = np.array(target_pos) - np.array(pocket_pos)
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return None
        
        direction = direction / distance
        aim_point = np.array(target_pos) + direction * (2 * self.BALL_RADIUS)
        return aim_point
    
    def _calculate_shot_angle(self, cue_pos, aim_point):
        """计算击球角度（phi）"""
        direction = np.array(aim_point) - np.array(cue_pos)
        angle_rad = np.arctan2(direction[1], direction[0])
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg
    
    def _calculate_shot_power(self, distance, target_to_pocket):
        """根据距离计算击球力度（优化版：降低力度减少白球落袋）"""
        total_dist = distance + target_to_pocket
        
        # 降低力度系数，减少白球落袋风险
        if total_dist < 0.4:
            power = 1.5  # 近距离轻打
        elif total_dist < 0.8:
            power = 2.2  # 中近距离
        elif total_dist < 1.2:
            power = 3.0  # 中距离
        elif total_dist < 1.8:
            power = 4.0  # 中远距离
        else:
            power = 4.8  # 远距离
        
        return power * self.POWER_REDUCTION
    
    def _check_path_clear(self, start_pos, end_pos, balls, exclude_ids):
        """检查路径上是否有障碍球"""
        line_vec = np.array(end_pos) - np.array(start_pos)
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return True
        
        line_vec = line_vec / line_length
        
        for ball_id, ball in balls.items():
            if ball_id in exclude_ids or ball.state.s == 4:
                continue
            
            ball_pos = self._get_ball_position(ball)
            to_ball = ball_pos - np.array(start_pos)
            projection = np.dot(to_ball, line_vec)
            
            if 0 < projection < line_length:
                closest_point = np.array(start_pos) + line_vec * projection
                distance = np.linalg.norm(ball_pos - closest_point)
                
                if distance < 2.2 * self.BALL_RADIUS:
                    return False
        return True
    
    def _get_first_contact_ball(self, cue_pos, phi, balls):
        """预测白球首先接触的球（精确版）"""
        # 计算击球方向
        phi_rad = np.radians(phi)
        direction = np.array([np.cos(phi_rad), np.sin(phi_rad)])
        
        candidates = []
        
        for ball_id, ball in balls.items():
            if ball_id == 'cue' or ball.state.s == 4:
                continue
            
            ball_pos = self._get_ball_position(ball)
            to_ball = ball_pos - np.array(cue_pos)
            
            # 投影到击球方向
            proj = np.dot(to_ball, direction)
            if proj <= 0:
                continue  # 球在后方
            
            # 计算垂直距离
            perp_dist = np.abs(np.cross(direction, to_ball))
            
            # 考虑碰撞半径（两球半径之和）
            collision_radius = 2 * self.BALL_RADIUS
            
            if perp_dist < collision_radius:
                # 计算实际碰撞点的距离
                # 使用勾股定理计算碰撞点
                if collision_radius**2 - perp_dist**2 > 0:
                    adjust = np.sqrt(collision_radius**2 - perp_dist**2)
                    collision_dist = proj - adjust
                    if collision_dist > 0:
                        candidates.append((collision_dist, ball_id))
        
        if not candidates:
            return None
        
        # 返回最近的球
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _will_hit_eight_ball(self, cue_pos, phi, balls, target_id):
        """检查是否会在击中目标球之前或之后击中8号球"""
        if '8' not in balls or balls['8'].state.s == 4:
            return False
        
        first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
        
        # 如果首球就是8号球，而目标不是8号球
        if first_contact == '8' and target_id != '8':
            return True
        
        # 如果首球不是目标球，可能会产生连锁碰撞到8号球
        if first_contact is not None and first_contact != target_id:
            # 这种情况下可能会犯规
            return True
        
        return False
    
    def _evaluate_shot_quality(self, cue_pos, target_id, target_pos, pocket_pos, balls, my_targets):
        """评估击球质量（综合评分）- 增强版"""
        score = 100.0
        
        # 计算瞄准点
        aim_point = self._calculate_aim_point(target_pos, pocket_pos)
        if aim_point is None:
            return -1000
        
        # 1. 距离因素
        cue_to_aim = self._calculate_distance(cue_pos, aim_point)
        target_to_pocket = self._calculate_distance(target_pos, pocket_pos)
        total_distance = cue_to_aim + target_to_pocket
        score -= total_distance * 15
        
        # 2. 角度因素（切角难度）
        vec1 = np.array(aim_point) - np.array(cue_pos)
        vec2 = np.array(pocket_pos) - np.array(target_pos)
        
        if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            cos_angle = np.dot(vec1, vec2)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            if angle < 20:
                score += 40  # 几乎直球，非常容易
            elif angle < 40:
                score += 20
            elif angle < 60:
                score += 0
            elif angle < 80:
                score -= 20
            else:
                score -= 50  # 大角度很难
        
        # 3. 路径清晰度检查
        # 白球到瞄准点
        if not self._check_path_clear(cue_pos, aim_point, balls, ['cue', target_id]):
            score -= 60
        
        # 目标球到袋口
        if not self._check_path_clear(target_pos, pocket_pos, balls, ['cue', target_id]):
            score -= 40
        
        # 4. 袋口距离奖励
        if target_to_pocket < 0.2:
            score += 50  # 非常接近袋口
        elif target_to_pocket < 0.4:
            score += 30
        elif target_to_pocket < 0.6:
            score += 10
        
        # 5. 首球检查（关键！）
        phi = self._calculate_shot_angle(cue_pos, aim_point)
        first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
        
        # 确定当前应该打什么
        remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
        should_hit_eight = len(remaining_own) == 0
        
        if first_contact is not None and first_contact != target_id:
            # 首球不是目标球
            if should_hit_eight:
                # 应该打8号球，但首球不是8号球
                if first_contact != '8':
                    score -= 250  # 严重犯规
            else:
                # 还有目标球，首球不是己方球
                if first_contact not in my_targets or first_contact == '8':
                    score -= 250  # 严重惩罚：首球犯规
        
        # 6. 黑8风险检查
        if target_id != '8' and '8' in [b for b in balls if balls[b].state.s != 4]:
            eight_pos = self._get_ball_position(balls['8'])
            # 检查击球路线是否经过黑8
            if not self._check_path_clear(cue_pos, aim_point, balls, ['cue', target_id, '8']):
                # 路线经过黑8附近
                dist_to_eight = self._calculate_distance(aim_point, eight_pos)
                if dist_to_eight < 3 * self.BALL_RADIUS:
                    score -= 150  # 有撞到黑8的风险
            
            # 额外检查：目标球到袋口的路线是否经过黑8
            if not self._check_path_clear(target_pos, pocket_pos, balls, ['cue', target_id, '8']):
                dist_eight_to_line = self._calculate_distance(eight_pos, pocket_pos)
                if dist_eight_to_line < 4 * self.BALL_RADIUS:
                    score -= 100  # 打进目标球可能连带打进黑8
        
        return score
    
    def _simulate_and_evaluate(self, action, balls, my_targets, table, add_noise=False):
        """模拟击球并详细评估结果（增强版：支持噪声模拟）
        
        参数：
            add_noise: 是否添加噪声模拟真实环境
        """
        try:
            sim_balls = self._clone_balls_for_sim(balls)
            # table 在仿真中应当是只读的；避免每次 deepcopy（同一 decision 会调用上百次）
            sim_table = table
            cue = pt.Cue(cue_ball_id="cue")

            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            # 如果启用噪声，添加高斯扰动
            if add_noise:
                noisy_action = {
                    'V0': action['V0'] + np.random.normal(0, self.noise_std['V0']),
                    'phi': action['phi'] + np.random.normal(0, self.noise_std['phi']),
                    'theta': action['theta'] + np.random.normal(0, self.noise_std['theta']),
                    'a': action['a'] + np.random.normal(0, self.noise_std['a']),
                    'b': action['b'] + np.random.normal(0, self.noise_std['b'])
                }
                # 限制参数在合理范围内
                noisy_action['V0'] = np.clip(noisy_action['V0'], 0.5, 8.0)
                noisy_action['phi'] = noisy_action['phi'] % 360
                noisy_action['theta'] = np.clip(noisy_action['theta'], 0, 90)
                noisy_action['a'] = np.clip(noisy_action['a'], -0.5, 0.5)
                noisy_action['b'] = np.clip(noisy_action['b'], -0.5, 0.5)
                actual_action = noisy_action
            else:
                actual_action = action
            
            shot.cue.set_state(
                V0=actual_action['V0'],
                phi=actual_action['phi'],
                theta=actual_action['theta'],
                a=actual_action['a'],
                b=actual_action['b']
            )
            
            if not simulate_with_timeout(shot, timeout=self.SIMULATION_TIMEOUT):
                return -200
            
            # 分析结果
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            score = 0
            
            # 确定当前应该打什么球
            remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            should_hit_eight = len(remaining_own) == 0
            
            # 严重犯规检查 - 白球落袋
            if 'cue' in new_pocketed:
                if '8' in new_pocketed:
                    return -10000  # 白球+黑8同时进袋，一票否决，绝对禁止
                return -1000  # 白球进袋（大幅提高惩罚，活着比进球重要）
            
            # 黑8处理 - 一票否决制
            if '8' in new_pocketed:
                if should_hit_eight:
                    return 250  # 合法打进黑8，获胜！
                else:
                    return -10000  # 误打黑8，一票否决，绝对禁止执行此动作
            
            # 首球/碰库判定：严格对齐 poolenv.py（重要：犯规会回滚，上面 pocketed 也不会被保留）
            first_contact_ball_id = None
            valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                    other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                    if other_ids:
                        first_contact_ball_id = other_ids[0]
                        break

            # 无碰撞：poolenv 会回滚并交换球权
            if first_contact_ball_id is None:
                return -900

            # 首球合法性：poolenv 判定为“碰到对手球或黑8（未清台）则犯规”
            remaining_own_before = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            if len(remaining_own_before) > 0:
                # 还有己方球，首球必须是己方球，且不能是 8
                if (first_contact_ball_id not in my_targets) or (first_contact_ball_id == '8'):
                    return -900
            else:
                # 只剩黑8
                if first_contact_ball_id != '8':
                    return -900

            # 未进球时必须碰库，否则 poolenv 会回滚
            if len(new_pocketed) == 0:
                cue_hit_cushion = False
                target_hit_cushion = False
                for e in shot.events:
                    et = str(e.event_type).lower()
                    ids = list(e.ids) if hasattr(e, 'ids') else []
                    if 'cushion' in et:
                        if 'cue' in ids:
                            cue_hit_cushion = True
                        if first_contact_ball_id is not None and first_contact_ball_id in ids:
                            target_hit_cushion = True
                if (not cue_hit_cushion) and (not target_hit_cushion):
                    return -650
            
            # 进球得分
            own_pocketed = [bid for bid in new_pocketed if bid in my_targets and bid != '8']
            enemy_pocketed = [bid for bid in new_pocketed if bid not in my_targets and bid not in ['cue', '8']]
            
            score += len(own_pocketed) * 100  # 提高进球奖励
            score -= len(enemy_pocketed) * 40
            
            # 若动作将交给对手（本杆未打进己方球），评估“留给对手的简单球”并惩罚
            own_pocketed = [bid for bid in new_pocketed if bid in my_targets and bid != '8']
            if len(own_pocketed) == 0 and '8' not in new_pocketed:
                threat = self._estimate_opponent_threat(shot.balls, sim_table, my_targets)
                # threat>0 表示对手存在清晰进球线路。阈值以上按强度惩罚。
                if threat > 5.0:
                    score -= min(260.0, (threat - 5.0) * 6.0)
            
            # 无进球但合法（碰库了）
            if score == 0 and 'cue' not in new_pocketed and '8' not in new_pocketed:
                score = 10
            
            # 袋口斥力场检测：白球停在袋口边缘是极度危险的
            if 'cue' not in new_pocketed and 'cue' in sim_balls:
                cue_final_pos = self._get_ball_position(sim_balls['cue'])
                pockets = self._get_pocket_positions_cached(sim_table)
                
                min_pocket_dist = float('inf')
                for pocket_pos in pockets:
                    dist = self._calculate_distance(cue_final_pos, pocket_pos)
                    if dist < min_pocket_dist:
                        min_pocket_dist = dist
                
                # 袋口危险区域：1.5倍球半径以内
                danger_radius = 1.5 * self.BALL_RADIUS
                if min_pocket_dist < danger_radius:
                    # 距离越近惩罚越重
                    danger_penalty = int(500 * (1 - min_pocket_dist / danger_radius))
                    score -= danger_penalty
                    logger.debug(f"[袋口斥力] 白球距袋口 {min_pocket_dist*1000:.1f}mm，扣除 {danger_penalty} 分")
            
            return score
            
        except Exception as e:
            logger.error(f"[NewAgent] 模拟失败: {e}")
            return -200
    
    def _evaluate_with_robustness(self, action, balls, my_targets, table, samples=None):
        """带噪声的鲁棒性评估（多次采样取平均/最小值）
        
        对同一动作进行多次带噪声模拟，评估其在真实环境中的可靠性。
        使用保守策略：如果有任何一次模拟出现严重犯规，大幅降低评分。
        """
        if samples is None:
            samples = self.ROBUSTNESS_SAMPLES_FINAL

        scores = []
        fatal_errors = []  # 记录致命错误的具体得分

        for i in range(samples):
            score = self._simulate_and_evaluate(action, balls, my_targets, table, add_noise=True)
            scores.append(score)
            
            # 统计致命错误（黑8误入、白球落袋）
            if score <= -5000:  # 一票否决级别的错误
                fatal_errors.append((i + 1, score))
                # 早停：一旦出现致命错误，立即拒绝该动作（大幅加速失败候选的筛除）
                # logger.warning(
                #     f"[鲁棒性检测] 动作在第{i + 1}/{samples}次模拟出现致命错误: {score}，提前拒绝"
                # )
                return -10000
        
        # 保守策略：使用平均分和最低分的加权组合
        # 这样既考虑平均表现，又惩罚高风险动作
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        
        # 60%平均分 + 40%最低分，更加保守（提高最低分权重）
        robust_score = avg_score * 0.6 + min_score * 0.4
        
        # 额外检查：如果有多次高风险错误（白球落袋、首球犯规等），大幅降低评分
        high_risk_errors = sum(1 for s in scores if s <= -500)  # 包含白球落袋(-1000)和袋口危险
        high_risk_threshold = max(2, int(math.ceil(samples * 0.2)))
        if high_risk_errors >= high_risk_threshold:
            # logger.warning(f"[鲁棒性检测] 动作有{high_risk_errors}/{samples}次高风险错误（≤-500），大幅降低评分")
            robust_score -= 300  # 大幅惩罚
        
        # 中等风险错误检测
        medium_risk_errors = sum(1 for s in scores if -500 < s <= -200)
        medium_risk_threshold = max(4, int(math.ceil(samples * 0.4)))
        if medium_risk_errors >= medium_risk_threshold:
            # logger.warning(f"[鲁棒性检测] 动作有{medium_risk_errors}/{samples}次中等错误（-500~-200），降低评分")
            robust_score -= 150
        
        return robust_score
    
    def _find_best_shot(self, balls, my_targets, table):
        """寻找最佳击球方案"""
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return None, -1000, None
        
        cue_pos = self._get_ball_position(cue_ball)
        pockets = self._get_pocket_positions(table)
        
        best_action = None
        best_score = -1000
        best_details = None

        # 第一阶段：对全部候选做“快速单次仿真”筛选（不加噪声），减少昂贵的鲁棒评估次数
        candidates = []
        
        # 确定实际目标球（排除已进袋的）
        active_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        for target_id in active_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            
            for pocket_pos in pockets:
                # 几何评估
                geo_score = self._evaluate_shot_quality(
                    cue_pos, target_id, target_pos, pocket_pos, balls, my_targets
                )
                
                if geo_score > best_score - 30:  # 只考虑较好的方案
                    aim_point = self._calculate_aim_point(target_pos, pocket_pos)
                    if aim_point is None:
                        continue
                    
                    phi = self._calculate_shot_angle(cue_pos, aim_point)
                    cue_to_aim = self._calculate_distance(cue_pos, aim_point)
                    target_to_pocket = self._calculate_distance(target_pos, pocket_pos)
                    V0 = self._calculate_shot_power(cue_to_aim, target_to_pocket)
                    
                    action = {
                        'V0': V0,
                        'phi': phi,
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    }

                    # 快速单次仿真（不加噪声）用于筛选
                    sim_score_fast = self._simulate_and_evaluate(action, balls, my_targets, table, add_noise=False)
                    if sim_score_fast <= -5000:
                        continue

                    total_fast = geo_score * self.GEO_WEIGHT + sim_score_fast * self.SIM_WEIGHT
                    candidates.append((total_fast, action, target_id, pocket_pos, geo_score, sim_score_fast))

        if not candidates:
            return None, -1000, None

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[: self.ROBUST_TOPK]

        # 第二阶段：仅对 Top-K 做全量鲁棒评估，保持最终精度
        for _, action, target_id, pocket_pos, geo_score, sim_score_fast in top_candidates:
            sim_score_robust = self._evaluate_with_robustness(
                action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_FINAL
            )
            total_score = geo_score * self.GEO_WEIGHT + sim_score_robust * self.SIM_WEIGHT

            if total_score > best_score:
                best_score = total_score
                best_action = action
                best_details = {
                    'target_id': target_id,
                    'pocket': pocket_pos,
                    'geo_score': geo_score,
                    'sim_score_fast': sim_score_fast,
                    'sim_score': sim_score_robust,
                }
        
        return best_action, best_score, best_details
    
    def _refine_shot_with_simulation(self, base_action, balls, my_targets, table):
        """使用物理模拟微调击球参数（增强版：更多安全检查）"""
        best_action = base_action.copy()
        # 微调阶段用更少采样做筛选（加速），最后对最优动作做全量采样复核（保精度）
        best_score_screen = self._evaluate_with_robustness(
            base_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_SCREEN
        )
        
        cue_ball = balls.get('cue')
        if cue_ball:
            cue_pos = self._get_ball_position(cue_ball)
        else:
            best_score_final = self._evaluate_with_robustness(
                best_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_FINAL
            )
            return best_action, best_score_final
        
        for i in range(self.SIMULATION_COUNT):
            # 微调参数（减小搜索范围以提高精度）
            test_action = {
                'V0': base_action['V0'] + np.random.uniform(-0.5, 0.5),
                'phi': base_action['phi'] + np.random.uniform(-4, 4),
                'theta': np.random.uniform(0, 12),
                'a': np.random.uniform(-0.12, 0.12),
                'b': np.random.uniform(-0.12, 0.12)
            }
            
            # 限制力度范围（更保守）
            test_action['V0'] = np.clip(test_action['V0'], 1.2, 5.5)
            test_action['phi'] = test_action['phi'] % 360
            test_action['theta'] = np.clip(test_action['theta'], 0, 15)
            
            # 快速首球检查（在模拟之前）
            first_contact = self._get_first_contact_ball(cue_pos, test_action['phi'], balls)
            remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            
            # 检查首球是否合法
            if len(remaining_own) > 0:
                # 还有目标球，首球必须是己方球
                if first_contact is not None and first_contact not in my_targets:
                    continue  # 跳过这个参数组合
                if first_contact == '8':
                    continue  # 不能先打8号球
            else:
                # 只剩8号球，首球必须是8号球
                if first_contact is not None and first_contact != '8':
                    continue
            
            # 使用筛选级鲁棒性评估（更少采样）
            score_screen = self._evaluate_with_robustness(
                test_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_SCREEN
            )
            
            if score_screen > best_score_screen:
                best_action = test_action.copy()
                best_score_screen = score_screen
                # logger.info(f"[NewAgent] 找到更优方案 (筛选鲁棒得分: {score_screen:.1f})")

        best_score_final = self._evaluate_with_robustness(
            best_action, balls, my_targets, table, samples=self.ROBUSTNESS_SAMPLES_FINAL
        )
        logger.info(
            f"[NewAgent] 模拟优化完成 ({self.SIMULATION_COUNT}次搜索), 最终鲁棒得分: {best_score_final:.1f}"
        )
        return best_action, best_score_final
    
    def _try_kick_shot(self, balls, my_targets, table, cue_pos, active_targets):
        """尝试大力解球（Kick Shot）策略
        
        当常规防守无法找到安全方案时，尝试用大力击球依靠复杂碰撞创造机会。
        关键：必须确保首球合法，即使后续轨迹复杂也要遵守规则。
        """
        logger.info("[NewAgent] 尝试大力解球策略（Kick Shot）")
        
        best_kick = None
        best_kick_score = -1000
        
        # 策略1：向最近的合法目标球大力击打，依靠多次碰撞和碰库
        for target_id in active_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            phi = self._calculate_shot_angle(cue_pos, target_pos)
            
            # 验证首球合法性
            first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
            if first_contact != target_id:
                continue  # 首球不合法，跳过
            
            # 大力击球：V0 = 5.0~6.5，制造复杂局面
            for v0 in [5.0, 5.5, 6.0, 6.5]:
                # 微调角度，避免直接进袋白球
                for delta_phi in [0, -3, 3, -6, 6]:
                    test_phi = (phi + delta_phi) % 360
                    test_first = self._get_first_contact_ball(cue_pos, test_phi, balls)
                    
                    if test_first != target_id:
                        continue  # 首球不合法
                    
                    kick_action = {
                        'V0': v0,
                        'phi': test_phi,
                        'theta': np.random.uniform(0, 8),  # 轻微跳球增加不可预测性
                        'a': 0.0,
                        'b': 0.0
                    }
                    
                    # 使用鲁棒性评估（但降低标准，只要不致命即可）
                    score = self._evaluate_with_robustness(kick_action, balls, my_targets, table)
                    
                    # 大力解球的接受标准：不出现致命错误即可
                    if score > -5000 and score > best_kick_score:
                        best_kick = kick_action
                        best_kick_score = score
                        logger.info(f"[Kick Shot] 找到可行方案: V0={v0:.1f}, phi={test_phi:.1f}, 得分={score:.1f}")
        
        # 策略2：如果直线大力不行，尝试碰库后击打（Bank Shot）
        if best_kick is None or best_kick_score < -200:
            logger.info("[NewAgent] 尝试碰库解球（Bank Shot）")
            # 尝试向球台边缘方向大力击打，利用碰库改变轨迹
            pockets = self._get_pocket_positions(table)
            for pocket_pos in pockets:
                # 计算朝向袋口附近（但不是直接瞄准）的角度
                direction = np.array(pocket_pos) - np.array(cue_pos)
                base_phi = np.degrees(np.arctan2(direction[1], direction[0]))
                if base_phi < 0:
                    base_phi += 360
                
                # 偏移角度确保会碰库
                for offset in [-25, -20, -15, 15, 20, 25]:
                    test_phi = (base_phi + offset) % 360
                    first_contact = self._get_first_contact_ball(cue_pos, test_phi, balls)
                    
                    # 检查首球合法性
                    if first_contact is not None and first_contact not in active_targets:
                        continue
                    
                    bank_action = {
                        'V0': 5.5,  # 中等偏大力度
                        'phi': test_phi,
                        'theta': 5.0,  # 轻微跳球
                        'a': 0.0,
                        'b': -0.1  # 轻微低杆，增加碰库后效果
                    }
                    
                    score = self._evaluate_with_robustness(bank_action, balls, my_targets, table)
                    
                    if score > -5000 and score > best_kick_score:
                        best_kick = bank_action
                        best_kick_score = score
                        logger.info(f"[Bank Shot] 找到可行方案: phi={test_phi:.1f}, 得分={score:.1f}")
                        break  # 找到一个就够了
        
        if best_kick is not None:
            logger.info(f"[NewAgent] 采用大力解球，预期得分: {best_kick_score:.1f}")
            return best_kick
        
        return None
    
    def _get_safe_shot(self, balls, my_targets, table):
        """生成安全的防守击球（增强版v2：设置防守底线，绝境时大力解球）"""
        SAFE_SHOT_THRESHOLD = -50  # 防守底线：低于此分数的动作视为不可接受
        
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return self._random_action()
        
        cue_pos = self._get_ball_position(cue_ball)
        
        # 确定实际目标（排除已进袋的）
        active_targets = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
        if len(active_targets) == 0:
            active_targets = ['8'] if balls.get('8') and balls['8'].state.s != 4 else []
        
        if not active_targets:
            return self._random_action()
        
        best_action = None
        best_score = -1000
        
        # 遍历所有目标球，找到能够安全击中的
        for target_id in active_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            
            # 计算直接指向目标球的角度
            phi = self._calculate_shot_angle(cue_pos, target_pos)
            
            # 预测首球
            first_contact = self._get_first_contact_ball(cue_pos, phi, balls)
            
            # 如果首球就是目标球，这是安全的
            if first_contact == target_id:
                # 验证这个击球 - 增加速度余量确保碰库
                dist_to_target = self._calculate_distance(cue_pos, target_pos)
                # 根据距离计算安全速度，确保有足够动能碰库
                safe_v0 = max(2.5, 1.5 + dist_to_target * 1.2)  # 最低2.5，加距离补偿
                safe_v0 = min(safe_v0, 4.0)  # 不要太大
                
                action = {
                    'V0': safe_v0,
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0
                }
                
                # 使用鲁棒性评估
                score = self._evaluate_with_robustness(action, balls, my_targets, table)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            else:
                # 首球不是目标球，尝试找其他角度
                # 尝试绕过障碍球的角度
                for delta_phi in [-5, 5, -10, 10, -15, 15, -20, 20, -30, 30]:
                    test_phi = (phi + delta_phi) % 360
                    test_first = self._get_first_contact_ball(cue_pos, test_phi, balls)
                    
                    if test_first == target_id:
                        # 增加速度余量确保碰库
                        dist_to_target = self._calculate_distance(cue_pos, target_pos)
                        safe_v0 = max(2.5, 1.5 + dist_to_target * 1.2)
                        safe_v0 = min(safe_v0, 4.0)
                        
                        action = {
                            'V0': safe_v0,
                            'phi': test_phi,
                            'theta': 0.0,
                            'a': 0.0,
                            'b': 0.0
                        }
                        # 使用鲁棒性评估
                        score = self._evaluate_with_robustness(action, balls, my_targets, table)
                        if score > best_score:
                            best_score = score
                            best_action = action
                            break  # 找到一个合法角度就停止
        
        # 如果还是没找到好的方案，尝试更多角度
        if best_action is None or best_score < -50:
            # 尝试所有可能的角度
            for test_phi in range(0, 360, 15):
                first_contact = self._get_first_contact_ball(cue_pos, test_phi, balls)
                
                if first_contact in active_targets:
                    action = {
                        'V0': 2.8,  # 增加速度确保碰库
                        'phi': test_phi,
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    }
                    # 使用鲁棒性评估
                    score = self._evaluate_with_robustness(action, balls, my_targets, table)
                    if score > best_score:
                        best_score = score
                        best_action = action
        
        if best_action is None:
            logger.warning("[NewAgent] 无法找到安全击球，使用最小风险策略")
            # 最后的方案：向最近的己方球轻轻一击
            min_dist = float('inf')
            nearest_target = None
            for tid in active_targets:
                if balls.get(tid) and balls[tid].state.s != 4:
                    dist = self._calculate_distance(cue_pos, self._get_ball_position(balls[tid]))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_target = tid
            
            if nearest_target:
                target_pos = self._get_ball_position(balls[nearest_target])
                phi = self._calculate_shot_angle(cue_pos, target_pos)
                dist_to_target = self._calculate_distance(cue_pos, target_pos)
                # 确保足够速度碰库
                safe_v0 = max(2.5, 1.5 + dist_to_target * 1.2)
                safe_v0 = min(safe_v0, 4.0)
                
                fallback_action = {
                    'V0': safe_v0,
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0
                }
                # 必须对后备方案进行鲁棒性评估，防止误打黑8
                fallback_score = self._evaluate_with_robustness(fallback_action, balls, my_targets, table)
                
                # 只有在评估通过时才使用这个后备方案
                if fallback_score > best_score:
                    best_action = fallback_action
                    best_score = fallback_score
                
                # 如果后备方案也失败（如误打黑8），返回随机动作
                if best_action is None or best_score <= -5000:
                    logger.warning("[NewAgent] 后备方案也不安全，使用随机动作")
                    return self._random_action()
            else:
                return self._random_action()
        
        # 防守底线检查：如果最佳防守得分太低（大概率犯规），尝试大力解球
        if best_action is not None and best_score < SAFE_SHOT_THRESHOLD:
            logger.warning(f"[NewAgent] 防守得分过低 ({best_score:.1f} < {SAFE_SHOT_THRESHOLD})，尝试大力解球策略")
            kick_action = self._try_kick_shot(balls, my_targets, table, cue_pos, active_targets)
            if kick_action is not None:
                return kick_action
            # 大力解球也失败，返回随机动作
            logger.warning("[NewAgent] 大力解球也未找到合法方案，使用随机动作")
            return self._random_action()
        
        # 最终安全检查：确保返回的动作是经过验证的
        if best_action is None or best_score <= -5000:
            logger.warning(f"[NewAgent] 所有安全策略都失败 (best_score={best_score:.1f})，尝试大力解球")
            kick_action = self._try_kick_shot(balls, my_targets, table, cue_pos, active_targets)
            if kick_action is not None:
                return kick_action
            return self._random_action()
        
        logger.info(f"[NewAgent] 使用安全击球策略，预期得分: {best_score:.1f}")
        return best_action
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策函数（增强版：更多安全验证）"""
        if balls is None or my_targets is None or table is None:
            logger.warning("[NewAgent] 缺少必要信息，使用随机动作")
            return self._random_action()
        
        try:
            # 开球局面：不值得跑完整在线搜索（会浪费大量时间）
            if self._is_break_state(balls, my_targets):
                logger.info("[NewAgent] 检测到开球局面，采用快速开球策略")
                return self._break_decision(balls, my_targets, table)

            # 检查是否需要打8号球
            remaining_own = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                logger.info("[NewAgent] 目标球已清空，切换到8号球")
            
            logger.info(f"[NewAgent] 开始决策，目标球: {my_targets}")
            
            # 第一步：几何计算 + 物理验证找最佳方案
            base_action, base_score, details = self._find_best_shot(balls, my_targets, table)
            
            if details:
                logger.info(f"[NewAgent] 最佳方案: 目标球{details['target_id']}, "
                           f"几何={details['geo_score']:.1f}, 模拟={details['sim_score']:.1f}")
            
            # 如果最佳方案得分太低，使用安全策略
            if base_action is None or base_score < -80:
                logger.info(f"[NewAgent] 未找到好方案 (得分: {base_score:.1f})，使用安全策略")
                return self._get_safe_shot(balls, my_targets, table)
            
            # 第二步：物理模拟微调
            refined_action, final_score = self._refine_shot_with_simulation(
                base_action, balls, my_targets, table
            )
            
            # 如果优化后得分仍然很低，使用安全策略
            if final_score < -80:
                logger.info(f"[NewAgent] 优化后得分仍低 ({final_score:.1f})，改用安全策略")
                return self._get_safe_shot(balls, my_targets, table)
            
            # 最终安全验证
            cue_ball = balls.get('cue')
            if cue_ball:
                cue_pos = self._get_ball_position(cue_ball)
                first_contact = self._get_first_contact_ball(cue_pos, refined_action['phi'], balls)
                
                # 检查首球是否合法
                if len(remaining_own) > 0:
                    # 还有目标球
                    if first_contact is not None and (first_contact not in my_targets or first_contact == '8'):
                        logger.warning(f"[NewAgent] 最终验证发现首球不合法 ({first_contact})，改用安全策略")
                        return self._get_safe_shot(balls, my_targets, table)
                else:
                    # 只打8号球
                    if first_contact is not None and first_contact != '8':
                        logger.warning(f"[NewAgent] 最终验证发现首球不是8号球 ({first_contact})，改用安全策略")
                        return self._get_safe_shot(balls, my_targets, table)
            
            logger.info(f"[NewAgent] 最终决策: V0={refined_action['V0']:.2f}, "
                       f"phi={refined_action['phi']:.2f}, theta={refined_action['theta']:.2f}")
            
            return refined_action
            
        except Exception as e:
            logger.error(f"[NewAgent] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()