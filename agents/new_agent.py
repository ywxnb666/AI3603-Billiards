import math
import pooltool as pt
import numpy as np
import copy
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
    # pooltool 的 event-based 模拟在极端情况下会产生非常多事件，导致单次 simulate 卡很久。
    # Windows 下没有 SIGALRM，我们用 max_events 做软上限，避免评测被拖死。
    max_events = 320

    if not (hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")):
        try:
            pt.simulate(shot, inplace=True, max_events=max_events)
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
            pt.simulate(shot, inplace=True, max_events=max_events)
            return True

        pt.simulate(shot, inplace=True, max_events=max_events)
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
    """NewAgent（重构版）：参考 BasicAgentPro 的“候选动作 + UCB 分配预算”框架，并做进攻优先的增强。

    设计目标（按优先级）：
    1) 先把球打进（提升进球率/进攻效率）
    2) 在不牺牲进球率的前提下提高抗噪（候选更丰富 + 带噪仿真）
    3) 最后才是防守/走位等二阶收益（只给轻量启发奖励）
    """

    def __init__(self, n_simulations: int = 60, c_puct: float = 1.35):
        super().__init__()

        # ========== 基本参数 ==========
        self.BALL_RADIUS = 0.028575
        self.n_simulations = int(n_simulations)
        self.c_puct = float(c_puct)

        # 仿真超时（Windows 无 SIGALRM 时会降级为 max_events 软上限）
        self.SIMULATION_TIMEOUT = 2

        # 与环境接近的噪声（在仿真中注入）
        # 注意：我们主要通过“带噪仿真评估”来选更稳的动作；候选生成也会做冗余覆盖。
        self.sim_noise = {
            'V0': 0.10,
            'phi': 0.15,
            'theta': 0.10,
            'a': 0.005,
            'b': 0.005,
        }

        # ========== 候选动作生成（比 Pro 更丰富） ==========
        self.MAX_CANDIDATES = 42
        # Pro 只做了 phi±0.5 的扰动；这里更细一点，同时也覆盖 V0 的噪声
        self.PHI_OFFSETS = (0.0, -0.35, 0.35, -0.7, 0.7, -1.2, 1.2, -1.8, 1.8)
        self.V0_SCALES = (0.88, 1.0, 1.12)
        self.V0_ADDS = (0.0, -0.25, 0.35)
        # a/b 是杆头偏移（旋转相关），噪声虽小但在某些局面会放大；这里生成多档幅度版本做鲁棒覆盖
        self.AB_SMALL = (0.0, 0.03, -0.03)
        self.AB_MED = (0.0, 0.06, -0.06, 0.10, -0.10)

        # 轻微抬杆角度（theta）候选：用于提供停球/回库/跟进的有限能力（以当前进球为主，不做大幅抬杆）
        self.THETA_SET = (0.0, 1.5, 3.0)

        # 袋口容差：用“袋口中心的侧向偏移”制造多条可进线路，提高进球率与抗噪
        self.POCKET_OFFSET_FACTORS = (0.0, 0.5, -0.5, 1.0, -1.0)

        # ========== 奖励函数增强（进攻优先 + 减少犯规） ==========
        # 白球靠近袋口：大概率导致下一杆/噪声下 scratch，这里显式惩罚
        self.CUE_POCKET_DANGER_RADIUS = 1.6 * self.BALL_RADIUS
        self.CUE_POCKET_DANGER_PENALTY_MAX = 240.0

        # 清台前黑8靠近袋口：容易“送清台”或噪声误入，这里显式惩罚
        self.BLACK8_DANGER_RADIUS = 2.6 * self.BALL_RADIUS
        self.BLACK8_DANGER_PENALTY_MAX = 260.0
        self.BLACK8_MOVE_TOWARDS_POCKET_PENALTY = 180.0
        self.BLACK8_CONTACT_PENALTY = 80.0

        # 二阶（次要）：轻量走位/连击启发奖励（不要喧宾夺主）
        self.FOLLOWUP_WEIGHT = 0.22
        self.FOLLOWUP_BONUS_CLIP = 35.0

        # 性能：袋口缓存
        self._pockets_cache_table_id = None
        self._pockets_cache = None

        logger.info("NewAgent（重构版）初始化完成")

    # =========================
    #  基础几何工具（中文注释）
    # =========================
    def _get_ball_position(self, ball):
        return np.array([ball.state.rvw[0][0], ball.state.rvw[0][1]], dtype=float)

    def _get_pocket_positions_cached(self, table):
        table_id = id(table)
        if self._pockets_cache_table_id != table_id or self._pockets_cache is None:
            self._pockets_cache_table_id = table_id
            pockets = []
            for pocket in table.pockets.values():
                pos = pocket.center
                pockets.append(np.array([pos[0], pos[1]], dtype=float))
            self._pockets_cache = pockets
        return self._pockets_cache

    def _min_dist_to_any_pocket(self, pos, table) -> float:
        pockets = self._get_pocket_positions_cached(table)
        best = float('inf')
        for p in pockets:
            d = float(np.linalg.norm(np.array(pos, dtype=float) - p))
            if d < best:
                best = d
        return float(best)

    def _calc_angle_degrees(self, v) -> float:
        return float(math.degrees(math.atan2(v[1], v[0])) % 360)

    def _action_key(self, action: dict):
        return (
            round(float(action.get('V0', 0.0)), 3),
            round(float(action.get('phi', 0.0)), 3),
            round(float(action.get('theta', 0.0)), 3),
            round(float(action.get('a', 0.0)), 4),
            round(float(action.get('b', 0.0)), 4),
        )

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """Ghost ball：从 (目标球, 袋口) 反推母球应到达的碰撞点。"""
        vec_obj_to_pocket = np.array(pocket_pos, dtype=float) - np.array(obj_pos, dtype=float)
        dist_obj_to_pocket = float(np.linalg.norm(vec_obj_to_pocket))
        if dist_obj_to_pocket < 1e-8:
            return None
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos, dtype=float) - unit_vec * (2.0 * float(self.BALL_RADIUS))
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos, dtype=float)
        dist_cue_to_ghost = float(np.linalg.norm(vec_cue_to_ghost))
        if dist_cue_to_ghost < 1e-8:
            return None
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return float(phi), float(dist_cue_to_ghost), ghost_pos

    def _check_path_clear(self, start_pos, end_pos, balls, exclude_ids):
        """线段遮挡的粗略检测：判断路径附近是否有球（用于快速筛掉明显被挡住的线路）。"""
        start = np.array(start_pos, dtype=float)
        end = np.array(end_pos, dtype=float)
        line_vec = end - start
        line_len = float(np.linalg.norm(line_vec))
        if line_len < 1e-8:
            return True
        dir_vec = line_vec / line_len

        # 经验阈值：路径附近 2.15R 内视为被挡（略松一点，避免误杀）
        clear_r = 2.15 * float(self.BALL_RADIUS)
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4:
                continue
            bp = self._get_ball_position(ball)
            to_ball = bp - start
            proj = float(np.dot(to_ball, dir_vec))
            if proj <= 0.0 or proj >= line_len:
                continue
            closest = start + dir_vec * proj
            d = float(np.linalg.norm(bp - closest))
            if d < clear_r:
                return False
        return True

    def _predict_first_contact_ball(self, cue_pos, phi, balls):
        """用几何近似预测白球沿 phi 方向的首碰球（用于候选快速合法性过滤）。"""
        phi_rad = float(np.radians(phi))
        direction = np.array([math.cos(phi_rad), math.sin(phi_rad)], dtype=float)
        candidates = []
        for bid, ball in balls.items():
            if bid == 'cue' or ball.state.s == 4:
                continue
            bp = self._get_ball_position(ball)
            to_ball = bp - np.array(cue_pos, dtype=float)
            proj = float(np.dot(to_ball, direction))
            if proj <= 0.0:
                continue
            perp_dist = float(abs(np.cross(direction, to_ball)))
            collision_r = 2.0 * float(self.BALL_RADIUS)
            if perp_dist >= collision_r:
                continue
            inside = collision_r * collision_r - perp_dist * perp_dist
            if inside <= 0.0:
                continue
            adjust = float(math.sqrt(inside))
            hit_dist = proj - adjust
            if hit_dist > 0.0:
                candidates.append((hit_dist, bid))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _estimate_followup_potential(self, balls_after, table, my_targets) -> float:
        """次要项：估计下一杆（若继续出杆）是否容易继续进球，避免完全不走位。"""
        try:
            cue_ball = balls_after.get('cue')
            if cue_ball is None or cue_ball.state.s == 4:
                return 0.0
            cue_pos = self._get_ball_position(cue_ball)
            pockets = self._get_pocket_positions_cached(table)

            remaining = [bid for bid in my_targets if bid in balls_after and balls_after[bid].state.s != 4]
            if not remaining:
                return 0.0

            best = -1e9
            for tid in remaining:
                tb = balls_after.get(tid)
                if tb is None or tb.state.s == 4:
                    continue
                tpos = self._get_ball_position(tb)
                for pocket_pos in pockets:
                    g = self._get_ghost_ball_target(cue_pos, tpos, pocket_pos)
                    if g is None:
                        continue
                    phi, d_cue, ghost = g
                    if not self._check_path_clear(cue_pos, ghost, balls_after, exclude_ids=['cue', tid]):
                        continue
                    if not self._check_path_clear(tpos, pocket_pos, balls_after, exclude_ids=['cue', tid]):
                        continue
                    d_obj = float(np.linalg.norm(np.array(pocket_pos, dtype=float) - np.array(tpos, dtype=float)))
                    # 只要一个粗略值：距离短/切角小 -> 分高
                    score = 80.0 - 18.0 * (d_cue + d_obj)
                    if score > best:
                        best = score
            return float(max(0.0, best)) if best > -1e8 else 0.0
        except Exception:
            return 0.0

    # =========================
    #  仿真与奖励（核心优化点）
    # =========================
    def _clone_balls_for_sim(self, balls):
        """轻量复制球对象：只 deep copy state，避免复制巨大 history（关键性能点）。"""
        if Ball is None or BallHistory is None:
            return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        sim_balls = {}
        empty_hist = BallHistory.factory
        for bid, ball in balls.items():
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

    def _simulate_action(self, balls, table, action: dict, add_noise: bool = True):
        """带噪声的物理仿真：返回 shot（失败返回 None）。"""
        sim_balls = self._clone_balls_for_sim(balls)
        sim_table = table  # table 视为只读，直接复用（比 deepcopy 快很多）
        cue = pt.Cue(cue_ball_id='cue')
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            if add_noise:
                noisy_V0 = float(np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0))
                noisy_phi = float((action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360)
                noisy_theta = float(np.clip(action.get('theta', 0.0) + np.random.normal(0, self.sim_noise['theta']), 0.0, 90.0))
                noisy_a = float(np.clip(action.get('a', 0.0) + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5))
                noisy_b = float(np.clip(action.get('b', 0.0) + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5))
            else:
                noisy_V0 = float(action['V0'])
                noisy_phi = float(action['phi']) % 360
                noisy_theta = float(action.get('theta', 0.0))
                noisy_a = float(action.get('a', 0.0))
                noisy_b = float(action.get('b', 0.0))

            shot.cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            ok = simulate_with_timeout(shot, timeout=int(self.SIMULATION_TIMEOUT))
            return shot if ok else None
        except Exception:
            return None

    def _analyze_shot_reward(self, shot: pt.System, last_state: dict, player_targets: list, table) -> float:
        """奖励函数：参考 Pro 的规则对齐计分，并增强“减少犯规概率”的项。

        主要增强：
        - 白球最终位置离袋口越近，越危险 -> 扣分
        - 清台前黑8离袋口越近/被推向袋口 -> 扣分
        - 走位/连击潜力 -> 轻量加分（次要）
        """
        # 1) 新进袋的球
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]

        # 2) 进球归属：黑8 只有在 player_targets=['8'] 时才算己方球
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ['cue', '8']]
        cue_pocketed = ('cue' in new_pocketed)
        eight_pocketed = ('8' in new_pocketed)

        # 3) 首球碰撞（从 events 中找 cue 与球的第一次接触）
        first_contact_ball_id = None
        valid_ball_ids = {str(i) for i in range(1, 16)}
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        foul_first_hit = False
        if first_contact_ball_id is None:
            # 未击中任何球：一般都算犯规（除非只剩黑8时，环境的特殊情况这里不强行放宽）
            foul_first_hit = True
        else:
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True

        # 4) 未进球必须碰库（白球或首碰球碰库即可）
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
        foul_no_rail = (len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion))

        # 5) 基础得分（保持 Pro 的温和尺度，利于 UCB）
        score = 0.0
        if cue_pocketed and eight_pocketed:
            score -= 500.0
        elif cue_pocketed:
            score -= 100.0
        elif eight_pocketed:
            is_targeting_eight_legally = (len(player_targets) == 1 and player_targets[0] == '8')
            score += 150.0 if is_targeting_eight_legally else -500.0

        if foul_first_hit:
            score -= 30.0
        if foul_no_rail:
            score -= 30.0

        score += float(len(own_pocketed)) * 50.0
        score -= float(len(enemy_pocketed)) * 20.0

        if score == 0.0 and (not cue_pocketed) and (not eight_pocketed) and (not foul_first_hit) and (not foul_no_rail):
            score = 10.0

        # 6) 增强项A：白球离袋口越近越危险（减少 scratch 概率）
        if (not cue_pocketed) and ('cue' in shot.balls) and (shot.balls['cue'].state.s != 4):
            try:
                cue_pos = self._get_ball_position(shot.balls['cue'])
                d = self._min_dist_to_any_pocket(cue_pos, table)
                if d < float(self.CUE_POCKET_DANGER_RADIUS):
                    ratio = 1.0 - float(d) / float(self.CUE_POCKET_DANGER_RADIUS)
                    score -= float(self.CUE_POCKET_DANGER_PENALTY_MAX) * float(np.clip(ratio, 0.0, 1.0))
            except Exception:
                pass

        # 7) 增强项B：清台前黑8风险（靠近袋口 / 被推向袋口 / 发生接触）
        if not (len(player_targets) == 1 and player_targets[0] == '8'):
            try:
                if '8' in shot.balls and shot.balls['8'].state.s != 4 and '8' in last_state and last_state['8'].state.s != 4:
                    eight_contact = False
                    for e in shot.events:
                        ids = list(e.ids) if hasattr(e, 'ids') else []
                        if '8' in ids and len(ids) >= 2:
                            eight_contact = True
                            break
                    if eight_contact:
                        score -= float(self.BLACK8_CONTACT_PENALTY)

                    eight_before = self._get_ball_position(last_state['8'])
                    eight_after = self._get_ball_position(shot.balls['8'])
                    d_before = self._min_dist_to_any_pocket(eight_before, table)
                    d_after = self._min_dist_to_any_pocket(eight_after, table)

                    if d_after < float(self.BLACK8_DANGER_RADIUS):
                        ratio = 1.0 - float(d_after) / float(self.BLACK8_DANGER_RADIUS)
                        score -= float(self.BLACK8_DANGER_PENALTY_MAX) * float(np.clip(ratio, 0.0, 1.0))

                    # 若明显更接近袋口：额外惩罚
                    delta = float(d_before) - float(d_after)
                    if delta > 0.015:
                        score -= float(self.BLACK8_MOVE_TOWARDS_POCKET_PENALTY) * float(np.clip(delta / 0.06, 0.0, 1.0))
            except Exception:
                pass

        # 8) 次要项：若本杆进球并继续出杆，给少量“下一杆可进球潜力”奖励
        if (len(own_pocketed) > 0) and (not cue_pocketed):
            try:
                follow = self._estimate_followup_potential(shot.balls, table, player_targets)
                score += float(self.FOLLOWUP_WEIGHT) * float(min(float(self.FOLLOWUP_BONUS_CLIP), follow))
            except Exception:
                pass

        return float(score)

    def _normalize_reward(self, raw_reward: float) -> float:
        """把 raw_reward 映射到 [0,1] 便于 UCB 稳定。"""
        lo, hi = -650.0, 350.0
        x = (float(raw_reward) - lo) / (hi - lo)
        return float(np.clip(x, 0.0, 1.0))

    # =========================
    #  候选生成（进球优先）
    # =========================
    def _estimate_v0_from_distance(self, dist_cue_to_ghost: float) -> float:
        """力度启发：距离越远力度越大（参考 Pro，但略更保守）。"""
        v = 1.45 + float(dist_cue_to_ghost) * 1.55
        return float(np.clip(v, 1.0, 6.0))

    def _geo_prior(self, cue_pos, obj_pos, pocket_pos, ghost_pos, balls, target_id: str) -> float:
        """几何先验分：用于给 UCB 一个初始排序（不参与回传，只用于截断候选）。"""
        # 路径被挡：强惩罚
        if not self._check_path_clear(cue_pos, ghost_pos, balls, exclude_ids=['cue', target_id]):
            return -1e9
        if not self._check_path_clear(obj_pos, pocket_pos, balls, exclude_ids=['cue', target_id]):
            return -1e9

        d1 = float(np.linalg.norm(np.array(ghost_pos, dtype=float) - np.array(cue_pos, dtype=float)))
        d2 = float(np.linalg.norm(np.array(pocket_pos, dtype=float) - np.array(obj_pos, dtype=float)))
        # 切角难度：用 (cue->ghost) 与 (obj->pocket) 的夹角
        v1 = np.array(ghost_pos, dtype=float) - np.array(cue_pos, dtype=float)
        v2 = np.array(pocket_pos, dtype=float) - np.array(obj_pos, dtype=float)
        cut = 90.0
        if float(np.linalg.norm(v1)) > 1e-8 and float(np.linalg.norm(v2)) > 1e-8:
            c = float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
            cut = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

        score = 100.0 - 18.0 * (d1 + d2) - 0.85 * cut
        # 目标球离袋口越近越好
        score += float(np.clip((0.65 - d2) * 50.0, -20.0, 35.0))
        return float(score)

    def _generate_candidate_actions(self, balls, my_targets, table):
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return []

        # 若已清台，强制只打黑8
        remaining = [bid for bid in my_targets if bid != '8' and bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ['8']

        target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if not target_ids:
            return []

        cue_pos = self._get_ball_position(cue_ball)
        pockets = self._get_pocket_positions_cached(table)

        scored = []
        for tid in target_ids:
            tb = balls.get(tid)
            if tb is None or tb.state.s == 4:
                continue
            obj_pos = self._get_ball_position(tb)

            for pocket_pos in pockets:
                # 袋口容差：对 pocket 做侧向偏移
                tp = np.array(pocket_pos, dtype=float) - np.array(obj_pos, dtype=float)
                tp_norm = float(np.linalg.norm(tp))
                if tp_norm < 1e-8:
                    continue
                tp_dir = tp / tp_norm
                perp = np.array([-tp_dir[1], tp_dir[0]], dtype=float)

                # 切角越大，允许的袋口偏移越小（避免产生假线路）
                base = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                if base is None:
                    continue
                _, _, ghost0 = base
                v1 = np.array(ghost0, dtype=float) - np.array(cue_pos, dtype=float)
                v2 = np.array(pocket_pos, dtype=float) - np.array(obj_pos, dtype=float)
                cut = 90.0
                if float(np.linalg.norm(v1)) > 1e-8 and float(np.linalg.norm(v2)) > 1e-8:
                    c = float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
                    cut = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
                if cut < 35.0:
                    pocket_offset = 0.90 * float(self.BALL_RADIUS)
                elif cut < 60.0:
                    pocket_offset = 0.70 * float(self.BALL_RADIUS)
                else:
                    pocket_offset = 0.50 * float(self.BALL_RADIUS)

                for f in self.POCKET_OFFSET_FACTORS:
                    pv = np.array(pocket_pos, dtype=float) + perp * float(f) * float(pocket_offset)
                    g = self._get_ghost_ball_target(cue_pos, obj_pos, pv)
                    if g is None:
                        continue
                    phi_ideal, dist_cue_to_ghost, ghost_pos = g

                    # 快速合法性过滤：首碰必须是目标球（几何预测）
                    first_contact = self._predict_first_contact_ball(cue_pos, phi_ideal, balls)
                    if first_contact != tid:
                        continue

                    # 几何先验分（用于截断）
                    geo = self._geo_prior(cue_pos, obj_pos, pv, ghost_pos, balls, tid)
                    if geo < -1e8:
                        continue

                    v_base = self._estimate_v0_from_distance(dist_cue_to_ghost)

                    # 根据几何难度自适应选择 a/b 与 theta 扰动档位
                    if (cut > 65.0) or (dist_cue_to_ghost > 1.10):
                        ab_list = self.AB_MED
                        theta_list = self.THETA_SET
                    elif (cut > 55.0) or (dist_cue_to_ghost > 0.95):
                        ab_list = self.AB_SMALL
                        theta_list = self.THETA_SET
                    else:
                        ab_list = (0.0,)
                        theta_list = (0.0,)

                    for dp in self.PHI_OFFSETS:
                        for scale in self.V0_SCALES:
                            for dv in self.V0_ADDS:
                                for a in ab_list:
                                    for b in ab_list:
                                        for theta in theta_list:
                                            act = {
                                                'V0': float(np.clip(v_base * float(scale) + float(dv), 0.8, 7.5)),
                                                'phi': float((phi_ideal + float(dp)) % 360),
                                                'theta': float(theta),
                                                'a': float(a),
                                                'b': float(b),
                                            }
                                            scored.append((float(geo), act))

                    # 额外注入“柔和安全”候选：降低力度与角度扰动幅度，提升合法性与抗噪（仅在困难线路时）
                    if (cut > 60.0 or dist_cue_to_ghost > 1.00) and geo < 85.0:
                        soft_v = float(np.clip(v_base * 0.85, 0.8, 6.5))
                        for dp_soft in (0.0, -0.25, 0.25):
                            act_soft = {
                                'V0': soft_v,
                                'phi': float((phi_ideal + float(dp_soft)) % 360),
                                'theta': 0.0,
                                'a': 0.0,
                                'b': 0.0,
                            }
                            scored.append((float(geo) - 4.0, act_soft))

        if not scored:
            return []

        # 去重 + 截断：先按几何分排序
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        candidates = []
        for geo, act in scored:
            k = self._action_key(act)
            if k in seen:
                continue
            seen.add(k)
            # 用 geo 做 prior（只用于排序/打印，可不参与 UCB）
            candidates.append({'action': act, 'prior': float(geo)})
            if len(candidates) >= int(self.MAX_CANDIDATES):
                break
        return candidates

    def _fallback_legal_action(self, balls, my_targets, table):
        """实在没有候选时的兜底：尽量保证首碰合法（优先朝最近目标球）。"""
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return self._random_action()

        remaining = [bid for bid in my_targets if bid != '8' and bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ['8']
        target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if not target_ids:
            return self._random_action()

        cue_pos = self._get_ball_position(cue_ball)
        # 选最近的目标球
        best_tid = None
        best_d = 1e9
        for tid in target_ids:
            d = float(np.linalg.norm(self._get_ball_position(balls[tid]) - cue_pos))
            if d < best_d:
                best_d = d
                best_tid = tid
        if best_tid is None:
            return self._random_action()

        tgt_pos = self._get_ball_position(balls[best_tid])
        base_phi = self._calc_angle_degrees(tgt_pos - cue_pos)

        # 小范围扫描，保证首碰是合法目标
        for dp in (0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -9.0, 9.0, -12.0, 12.0):
            phi = float((base_phi + dp) % 360)
            fc = self._predict_first_contact_ball(cue_pos, phi, balls)
            if fc == best_tid:
                return {'V0': 2.8, 'phi': phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0}

        return {'V0': 2.8, 'phi': float(base_phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0}

    # =========================
    #  主决策：Pro 风格 UCB
    # =========================
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        try:
            # 若已清台，强制只打黑8
            remaining = [bid for bid in my_targets if bid != '8' and bid in balls and balls[bid].state.s != 4]
            if len(remaining) == 0:
                my_targets = ['8']

            # 生成候选
            candidates = self._generate_candidate_actions(balls, my_targets, table)
            if not candidates:
                return self._fallback_legal_action(balls, my_targets, table)

            candidate_actions = [c['action'] for c in candidates]
            n = len(candidate_actions)
            N = np.zeros(n, dtype=np.float64)
            Q = np.zeros(n, dtype=np.float64)

            # 保存击球前快照（奖励函数需要对比进袋）
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            # UCB 预算分配：每次只仿真一个动作（带噪声），更贴近真实执行
            for i in range(int(self.n_simulations)):
                if i < n:
                    idx = int(i)
                else:
                    total_n = float(np.sum(N))
                    ucb = (Q / (N + 1e-6)) + float(self.c_puct) * np.sqrt(np.log(total_n + 1.0) / (N + 1e-6))
                    idx = int(np.argmax(ucb))

                shot = self._simulate_action(balls, table, candidate_actions[idx], add_noise=True)
                if shot is None:
                    raw_reward = -500.0
                else:
                    raw_reward = self._analyze_shot_reward(shot, last_state_snapshot, my_targets, table)

                rew = self._normalize_reward(raw_reward)
                N[idx] += 1.0
                Q[idx] += float(rew)

            avg = Q / (N + 1e-6)
            best_idx = int(np.argmax(avg))
            best_action = candidate_actions[best_idx]
            logger.info(f"[NewAgent] Best Avg={avg[best_idx]:.3f} Sims={self.n_simulations} V0={best_action['V0']:.2f} phi={best_action['phi']:.2f}")
            return best_action

        except Exception as e:
            logger.error(f"[NewAgent] 决策异常: {e}")
            return self._random_action()