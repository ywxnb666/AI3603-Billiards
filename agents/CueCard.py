import numpy as np
import pooltool as pt
import copy
import random
import math
from sklearn.cluster import KMeans
from datetime import datetime
import logging
import os
import warnings

# 假设 Agent 基类定义 (如果实际项目中在其他位置，请修改导入)
try:
    from .agent import Agent
except ImportError:
    class Agent:
        def decision(self, *args, **kwargs): pass


def _get_cuecard_logger() -> logging.Logger:
    """Create (or reuse) a CueCard logger that writes to logs/poolenv.log and terminal.

    Important: do NOT use logger name 'poolenv'. The environment module uses that
    name (via __name__), and overriding its handlers would accidentally silence
    PoolEnv/BasicAgentPro outputs.
    """
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'poolenv.log')

    logger = logging.getLogger('cuecard')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    has_file = False
    has_stream = False
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(log_path):
                    has_file = True
            except Exception:
                pass
        if isinstance(h, logging.StreamHandler):
            # FileHandler is also a StreamHandler subclass; exclude it.
            if not isinstance(h, logging.FileHandler):
                has_stream = True

    if not has_file:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger

class CueCardAgent(Agent):
    """
    CueCard 架构完整复刻版 (Single Machine Version)
    
    架构特征:
    1. Candidate Generation: 生成直线、翻袋、反弹、组合球。
    2. Level 1 Search (Noisy): 对候选进行带噪声的蒙特卡洛模拟。
    3. State Clustering: 将 L1 的大量结果状态聚类为 '代表性状态' (Representative States)。
    4. Level 2 Search: 对代表性状态进行更深层的推演 (Endgame时使用无噪搜索)。
    5. Scoring: 使用 Score = 1.0*p0 + 0.33*p1 ... 的概率评估函数。
    """

    def __init__(self, n_l1_sims=10, n_clusters=5, n_l2_sims=10):
        super().__init__()
        self.logger = _get_cuecard_logger()
        # 核心参数
        self.n_l1_sims = n_l1_sims      # 一级搜索模拟次数 (CueCard paper: 25-100)
        self.n_clusters = n_clusters    # 聚类数量 (CueCard paper: 5-10)
        self.n_l2_sims = n_l2_sims      # 二级搜索模拟次数
        
        # 物理参数
        self.ball_radius = 0.028575
        self.pocket_radius = 0.05       # 估算值
        
        # 噪声模型 (与 BasicAgentPro 对齐或根据环境设定)
        self.noise_std = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

        self.logger.info("[CueCard] 代理初始化完成")

    # =========================================================================
    # Phase 1: Candidate Generation (候选生成)
    # =========================================================================

    def _generate_bank_shots(self, cue_pos, obj_pos, table, tid, balls, candidates):
        """
        生成单库翻袋球 (One-rail Bank Shots)
        原理：将袋口关于库边做镜像，目标球瞄准镜像袋口。
        """
        # 获取球台边界 (假设 table中心在 0,0)
        # pooltool table specs: table.w 是宽(短边), table.l 是长(长边)
        w, l = table.w, table.l 
        rails = [
            {'name': 'right', 'axis': 0, 'val': w/2},
            {'name': 'left',  'axis': 0, 'val': -w/2},
            {'name': 'top',   'axis': 1, 'val': l/2},
            {'name': 'bottom','axis': 1, 'val': -l/2}
        ]

        for pocket in table.pockets.values():
            p_pos = pocket.center
            
            for rail in rails:
                # 1. 计算袋口的镜像位置
                mirrored_pocket = list(p_pos)
                axis = rail['axis']
                # 镜像公式: x' = 2*line - x
                mirrored_pocket[axis] = 2 * rail['val'] - p_pos[axis]
                
                # 2. 计算鬼球点 (针对镜像袋口)
                # 目标球 -> 镜像袋口 的向量
                vec_obj_mirror = np.array(mirrored_pocket) - np.array(obj_pos)
                dist = np.linalg.norm(vec_obj_mirror)
                if dist == 0: continue
                
                # 鬼球位置 (真实空间)
                # 注意：鬼球点是在目标球旁边，指向镜像袋口的方向
                ghost_pos = np.array(obj_pos) - (vec_obj_mirror / dist) * (2 * self.ball_radius)

                # [新增] 关键阻挡检测
                # 1. 母球 -> Ghost Ball (瞄准点) 是否有阻挡？
                if self._is_path_blocked(cue_pos, ghost_pos, balls, exclude=[tid]):
                    continue
                    
                # 2. 目标球 -> 镜像袋口 (实际行进路线) 是否有阻挡？
                # 注意：这里只检查目标球出发的一段，镜像点很远，主要检查中间有没有球挡路
                # 简单起见，检查 目标球 -> 库边撞击点 也可以，但镜像法其实等价于检查 目标球->镜像袋口
                # 我们排除掉 tid 自己
                if self._is_path_blocked(obj_pos, mirrored_pocket, balls, exclude=[tid]):
                    continue
                
                # 3. 检查反弹点是否在库边有效范围内 (简单裁剪)
                # 计算目标球到镜像袋口的连线与库边的交点
                t = (rail['val'] - obj_pos[axis]) / (mirrored_pocket[axis] - obj_pos[axis])
                if not (0 <= t <= 1): continue # 交点不在路径段上，虽一般不会发生，保险起见
                
                # 4. 计算母球击打参数
                phi, cue_dist = self._get_aim_params(cue_pos, ghost_pos, ghost_pos) # 目标即鬼球
                
                # 5. 过滤掉明显无法击打的角度 (例如反切)
                # 简单判断: 母球到鬼球的向量 与 鬼球到目标球向量 的夹角
                vec_cue_ghost = ghost_pos - np.array(cue_pos)
                angle_diff = abs(math.degrees(np.arctan2(vec_cue_ghost[1], vec_cue_ghost[0]) - 
                                              np.arctan2(vec_obj_mirror[1], vec_obj_mirror[0])))
                angle_diff = angle_diff % 360
                if angle_diff > 90 and angle_diff < 270: continue # 切球角度过大

                h_prob = self._calculate_heuristic_prob(cue_pos, obj_pos, mirrored_pocket, balls, tid, is_bank=True)
                # 添加候选 (力度通常需要大一点)
                for v in [4.0, 6.5]:
                    candidates.append({
                        'V0': v, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0, 
                        'type': 'bank', 'target': tid, 'h_prob': h_prob
                    })

    def _generate_kick_shots(self, cue_pos, obj_pos, table, tid, candidates):
        """
        生成单库反弹球 (Kick Shots) 并计算概率
        原理：将母球关于库边做镜像，计算撞击点。
        """
        w, l = table.w, table.l 
        rails = [
            {'name': 'right', 'axis': 0, 'val': w/2},
            {'name': 'left',  'axis': 0, 'val': -w/2},
            {'name': 'top',   'axis': 1, 'val': l/2},
            {'name': 'bottom','axis': 1, 'val': -l/2}
        ]
        
        # 遍历每一个袋口寻找可能的进球路径
        for pocket in table.pockets.values():
            p_pos = pocket.center
            
            # 1. 计算目标球进袋所需的鬼球点 (Ghost Ball)
            vec_obj_pocket = np.array(p_pos) - np.array(obj_pos)
            dist_obj_pocket = np.linalg.norm(vec_obj_pocket)
            if dist_obj_pocket == 0: continue
            
            # 鬼球位置: 目标球进袋时，母球应该在的位置
            ghost_pos = np.array(obj_pos) - (vec_obj_pocket / dist_obj_pocket) * (2 * self.ball_radius)
            
            # 检查【目标球 -> 袋口】是否阻挡
            if self._is_path_blocked(obj_pos, p_pos, {}, exclude=[tid]): continue # 这里 balls传空字典或只传关键障碍，简化计算

            # 2. 遍历每一个库边，尝试 Kick
            for rail in rails:
                axis = rail['axis']
                
                # 计算母球镜像位置
                mirrored_cue = list(cue_pos)
                mirrored_cue[axis] = 2 * rail['val'] - cue_pos[axis]
                mirrored_cue = np.array(mirrored_cue)
                
                # 向量: 镜像母球 -> 鬼球 (这就是母球撞库后的虚拟直线路径)
                vec_mirror_to_ghost = ghost_pos - mirrored_cue
                
                # 计算撞击库边的点 (Impact Point)
                # 利用相似三角形原理: (Impact - Mirror) / (Ghost - Mirror) = t
                if (ghost_pos[axis] - mirrored_cue[axis]) == 0: continue
                t = (rail['val'] - mirrored_cue[axis]) / (ghost_pos[axis] - mirrored_cue[axis])
                
                # 交点必须在线段上 (0 < t < 1) 且指向前方
                if not (0.01 < t < 0.99): continue
                
                impact_point = mirrored_cue + t * vec_mirror_to_ghost
                
                # 检查撞击点是否在有效库边长度内
                rail_len = l if axis == 0 else w
                other_axis = 1 - axis
                # 稍微留点余量，不要打到库角里去
                if abs(impact_point[other_axis]) > (rail_len / 2 - self.pocket_radius): continue
                
                # 3. 关键阻挡检测 (分两段)
                # Segment A: 母球 -> 库边撞击点 (排除 tid，因为是去打它)
                if self._is_path_blocked(cue_pos, impact_point, {}, exclude=[tid]): continue
                # Segment B: 库边撞击点 -> 鬼球点
                if self._is_path_blocked(impact_point, ghost_pos, {}, exclude=[tid]): continue
                
                # 4. 计算击球参数
                # 瞄准角度 = 镜像母球指向鬼球的角度
                phi = math.degrees(math.atan2(vec_mirror_to_ghost[1], vec_mirror_to_ghost[0])) % 360
                
                # 5. 计算概率 h_prob
                # 距离 = 母球到库 + 库到鬼球 + 鬼球到袋
                dist_cue_rail = np.linalg.norm(impact_point - cue_pos)
                dist_rail_ghost = np.linalg.norm(ghost_pos - impact_point)
                total_dist = dist_cue_rail + dist_rail_ghost + dist_obj_pocket
                
                # 切球角度 (Cut Angle)
                vec_rail_ghost = ghost_pos - impact_point
                try:
                    angle = pt.utils.angle(vec_rail_ghost, vec_obj_pocket)
                    angle_deg = math.degrees(angle)
                except: angle_deg = 90
                
                if angle_deg >= 80: continue # 太薄了，Kick很难控制薄球
                
                # 概率公式 (与直球类似，但增加难度惩罚)
                base_prob = (1.0 - angle_deg/90.0) * (1.0 / (1.0 + 0.5 * total_dist))
                h_prob = base_prob * 0.4 # [关键] Kick Shot 难度系数 0.4
                
                # 生成候选 (Kick 通常需要较大力度来保证反弹后的精度和动能)
                for v in [3.5, 5.0, 7.0]:
                    candidates.append({
                        'V0': v, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0, 
                        'type': 'kick', 'target': tid, 
                        'h_prob': h_prob
                    })

    def _generate_combination_shots(self, cue_pos, balls, table, my_targets, candidates):
        """
        生成组合球 (Combo): 母球 -> 中间球(Ball A) -> 目标球(Ball B) -> 袋口
        策略限制：为了保证进球有效，Ball B 必须是我的目标球或者黑8（如果是打8阶段）。
        Ball A 必须是我的目标球（规则要求首触必须是己方球）。
        """
        # 1. 确定首触球 (Ball A) - 必须是合法的己方目标球
        valid_first_balls = [bid for bid in my_targets if bid in balls]
        
        # 2. 确定进袋球 (Ball B) - 最好也是己方球，或者是8号球
        # 如果打进对方球是给对方加分，所以我们通常只考虑打进自己的球
        valid_second_balls = valid_first_balls + (['8'] if '8' in balls else [])
        
        for first_id in valid_first_balls:
            pos_a = balls[first_id].state.rvw[0]
            
            for second_id in valid_second_balls:
                if first_id == second_id: continue # 不能是同一个球
                
                pos_b = balls[second_id].state.rvw[0]
                
                # 距离剪枝：如果两个球相距太远(超过半张台)，组合球成功率极低，跳过
                if np.linalg.norm(pos_a - pos_b) > table.l / 2: continue

                for pid, pocket in table.pockets.items():
                    p_pos = pocket.center
                    
                    # --- 倒推法计算 ---
                    
                    # Step 1: Ball B -> 袋口
                    # 计算 Ball B 进袋所需的受力点 (Ghost Ball for B)
                    vec_b_pocket = p_pos - pos_b
                    dist_b_p = np.linalg.norm(vec_b_pocket)
                    if dist_b_p == 0: continue
                    unit_b_p = vec_b_pocket / dist_b_p
                    ghost_b = pos_b - unit_b_p * (2 * self.ball_radius)
                    
                    # Step 2: Ball A -> Ghost B
                    # Ball A 必须撞击 Ball B 的 Ghost_B 位置
                    vec_a_gb = ghost_b - pos_a
                    dist_a_gb = np.linalg.norm(vec_a_gb)
                    if dist_a_gb == 0: continue
                    unit_a_gb = vec_a_gb / dist_a_gb
                    
                    # 角度检查1: Ball A 撞击 Ball B 的角度是否合理 (不能太薄)
                    # Ball A 的行进方向 (unit_a_gb) 与 B->Pocket 方向 (unit_b_p) 的夹角
                    dot_check_1 = np.dot(unit_a_gb, unit_b_p)
                    if dot_check_1 < 0.2: continue # 角度过大，难以传力
                    
                    # 计算 Ball A 撞击 Ball B 时，Ball A 应该在的位置 (Ghost Ball for A)
                    ghost_a = pos_a - unit_a_gb * (2 * self.ball_radius) # 这里的计算其实是指向 Ghost_B 的反向
                    # 修正: 我们需要的是 "母球撞击 Ball A 的位置"
                    # Ball A 需要到达的位置其实就是 ghost_b 的接触瞬间位置。
                    # 所以 Ball A 被撞击时，母球应该瞄准 Ball A 的 "Ghost A"
                    # Ghost A 是为了让 Ball A 沿 vec_a_gb 运动
                    ghost_for_cue = pos_a - unit_a_gb * (2 * self.ball_radius)
                    
                    # Step 3: 母球 -> Ghost A
                    phi, dist_c_ga = self._get_aim_params(cue_pos, ghost_for_cue, ghost_for_cue)
                    
                    # 角度检查2: 母球切 Ball A 的角度
                    vec_c_ga = ghost_for_cue - cue_pos
                    if np.linalg.norm(vec_c_ga) == 0: continue
                    unit_c_ga = vec_c_ga / np.linalg.norm(vec_c_ga)
                    dot_check_2 = np.dot(unit_c_ga, unit_a_gb)
                    if dot_check_2 < 0: continue # 反角，打不到
                    # [新增] 组合球概率估算
                    # 简化：将第一球(pos_a)视作母球，第二球(pos_b)视作目标球，计算第二段的概率
                    prob_part2 = self._calculate_heuristic_prob(pos_a, pos_b, p_pos, balls, second_id)
                    h_prob = prob_part2 * 0.5 # 组合球难度极大，直接打5折
                    
                    # 添加到候选
                    candidates.append({
                        'V0': 4.5, # 组合球通常需要较大力度
                        'phi': phi, 'theta': 0, 'a': 0, 'b': 0,
                        'type': 'combo', 'target': second_id, 'h_prob': h_prob
                    })

    def _play_safety(self, balls, my_targets, table):
        safety_candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        targets = [b for b in my_targets if b in balls]
        if not targets: targets = ['8']

        # 1. 生成更广泛的防守候选
        for tid in targets:
            obj_pos = balls[tid].state.rvw[0]
            # 获取瞄准球心的角度
            base_phi, _ = self._get_aim_params(cue_pos, obj_pos, obj_pos)
            
            # 扩大搜索范围：从正面撞击到极薄的切球
            # -85 到 +85 度，每隔 5 度试一次
            for angle_offset in range(-85, 86, 5): 
                # 尝试多种力度，特别是极轻的力度
                for v in [0.5, 1.0, 2.0, 3.0]: 
                    safety_candidates.append({
                        'V0': v, 
                        'phi': (base_phi + angle_offset) % 360, 
                        'theta': 0, 'a': 0, 'b': 0,
                        'type': 'safety'
                    })
        
        # 如果还是没有（极罕见），尝试打向任意方向（避免 Random Action 的盲目）
        if not safety_candidates:
            self.logger.info("[CueCard] 安全模式启动.")
            for deg in range(0, 360, 10):
                safety_candidates.append({'V0': 1.0, 'phi': deg, 'theta':0, 'a':0, 'b':0})

        # 2. 评估防守效果 (保持原有逻辑，但增加对犯规的极严厉惩罚)
        best_safety_action = None
        best_safety_score = -float('inf')
        opp_targets = self._get_opponent_targets(my_targets)
        
        # 随机抽样 50 个候选进行评估，防止超时
        random.shuffle(safety_candidates)
        
        for action in safety_candidates[:50]:
            shot = self._simulate_shot(balls, table, action, noise=False)
            if shot:
                is_foul, _, _ = self.analyze_shot_result(shot, balls, my_targets)
                if is_foul:
                    continue # 绝对不要犯规

                score = self._calculate_safety_score(shot, opp_targets)
                if score > best_safety_score:
                    best_safety_score = score
                    best_safety_action = action
        
        if best_safety_action:
            self.logger.info(f"[CueCard] 安全模式. 分数: {best_safety_score:.2f}")
            return best_safety_action
            
        # 实在没办法，只能随机，但力度要小
        return {'V0': 0.5, 'phi': random.uniform(0,360), 'theta':0, 'a':0, 'b':0}

    def _calculate_safety_score(self, shot, opp_targets):
        """
        计算防守分数：母球距离对手所有目标球越远越好。
        """
        final_cue_pos = shot.balls['cue'].state.rvw[0]
        dists = []
        
        for opp_id in opp_targets:
            if opp_id in shot.balls:
                op_pos = shot.balls[opp_id].state.rvw[0]
                dists.append(np.linalg.norm(final_cue_pos - op_pos))
        
        if not dists: return 0
        
        # 策略：最大化"最近对手球的距离" (Max-Min Strategy)
        # 即：我不希望对手有任何简单的近球可打
        min_dist = min(dists)
        
        # 额外奖励：贴库 (如果母球贴库，对手很难处理)
        # 简单判断：母球中心离边界 < 半径 + 少量余量
        table_w, table_l = shot.table.w, shot.table.l
        is_rail = (abs(final_cue_pos[0]) > table_w/2 - 0.05) or \
                  (abs(final_cue_pos[1]) > table_l/2 - 0.05)
        
        score = min_dist + (0.5 if is_rail else 0)
        return score
    

    def _filter_targets_heuristically(self, balls, target_ids, cue_pos, table):
        """
        智能预筛选目标球：计算综合难度分 (角度 + 距离 + 阻挡)，只返回最容易打的几个球。
        防止对极难的球生成大量无效候选，浪费计算资源。
        """
        scored_targets = []
        
        for tid in target_ids:
            ball = balls[tid]
            obj_pos = ball.state.rvw[0]
            
            # 初始化该球的最低难度为无穷大
            best_difficulty = float('inf')
            
            # 遍历所有袋口，看打进这个袋口的最低难度
            for pocket in table.pockets.values():
                p_pos = pocket.center
                
                # --- A. 几何计算 ---
                # 1. 目标球 -> 袋口 向量
                vec_obj_pocket = np.array(p_pos) - np.array(obj_pos)
                dist_obj_pocket = np.linalg.norm(vec_obj_pocket)
                if dist_obj_pocket == 0: continue
                
                # 2. 鬼球点 (Ghost Ball Position)
                unit_obj_pocket = vec_obj_pocket / dist_obj_pocket
                ghost_pos = np.array(obj_pos) - unit_obj_pocket * (2 * self.ball_radius)
                
                # 3. 母球 -> 鬼球点 向量
                vec_cue_ghost = ghost_pos - np.array(cue_pos)
                dist_cue_ghost = np.linalg.norm(vec_cue_ghost)
                if dist_cue_ghost == 0: continue
                unit_cue_ghost = vec_cue_ghost / dist_cue_ghost
                
                # --- B. 评分因子 ---
                
                # 1. 切球角度 (Cut Angle)
                # 计算 母球行进方向(unit_cue_ghost) 与 进球方向(unit_obj_pocket) 的夹角
                # cos_angle 接近 1 (0度) 最好，接近 0 (90度) 最差
                cos_angle = np.dot(unit_cue_ghost, unit_obj_pocket)
                # 限制数值范围防止 arccos 报错
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                
                # 过滤掉无法击打的角度 (> 75度)
                if angle_deg > 75: 
                    difficulty = 10000 # 极难
                else:
                    # 2. 距离因子 (Distance Factor)
                    # 总距离越短越好，但 母球->目标球 的距离权重略高(越远越不准)
                    weighted_dist = dist_cue_ghost * 1.5 + dist_obj_pocket * 1.0
                    
                    # 3. 角度因子 (Angle Factor)
                    # 角度越大，难度指数级上升
                    angle_penalty = (angle_deg ** 1.5) / 10.0
                    
                    # 4. 阻挡检测 (Blocking Penalty) - 关键！
                    # 检查 母球->鬼球 路径 (排除目标球自身 tid)
                    is_cue_blocked = self._is_path_blocked(cue_pos, ghost_pos, balls, exclude=[tid])
                    # 检查 目标球->袋口 路径
                    is_obj_blocked = self._is_path_blocked(obj_pos, p_pos, balls, exclude=[tid])
                    
                    block_penalty = 0
                    if is_cue_blocked: block_penalty += 5000 # 被阻挡很难打
                    if is_obj_blocked: block_penalty += 5000 
                    
                    # 综合评分: 距离 + 角度 + 阻挡
                    difficulty = weighted_dist * 10 + angle_penalty + block_penalty

                if difficulty < best_difficulty:
                    best_difficulty = difficulty
            
            scored_targets.append((tid, best_difficulty))
            
        # 按难度排序 (分数越低越好)
        scored_targets.sort(key=lambda x: x[1])
        
        # 只保留最容易打的 4 个球进入候选生成
        # 如果所有球都很难(分都很高)，也会选出相对容易的去尝试(比如解球)
        primary_targets = [x[0] for x in scored_targets[:4]]
        
        # self.logger.info(
        #     f"[Heuristic] 预筛选目标球: 从 {len(target_ids)} 个减少到 {len(primary_targets)} 个: {primary_targets}"
        # )
        return primary_targets
    
    def _is_triangle_formation(self, balls):
        """
        检测是否摆好球（15颗球聚集成三角形）。
        判据：
        1. 台面上必须正好有 15 颗目标球（加母球共16颗）。
        2. 这 15 颗球必须紧密聚集在一起（所有球到质心的距离都很小）。
        """
        # 1. 提取所有目标球（非母球）
        obj_balls = [b for k, b in balls.items() if k != 'cue']
        
        # 2. 数量必须严格是 15
        if len(obj_balls) != 15:
            return False
            
        # 提取坐标数组
        positions = np.array([b.state.rvw[0] for b in obj_balls])
        
        # 3. 计算质心 (Center of Mass)
        centroid = np.mean(positions, axis=0)
        
        # 4. 计算紧密度 (所有球到质心的距离)
        distances = np.linalg.norm(positions - centroid, axis=1)
        
        # 5. 判定阈值
        # 标准 8 球摆法是 5 排，整体宽度约 4-5 个球直径。
        # 球半径 r ≈ 0.0285m，直径 d ≈ 0.057m。
        # 整个三角形的外接圆半径大约在 0.15m 左右。
        # 我们设定一个宽松的阈值 0.25m (25厘米)。
        # 如果所有 15 颗球都在质心周围 25cm 半径内，说明它们摆成了堆。
        # 如果球散开了，这个最大距离通常会远大于 0.25m。
        if np.max(distances) < 0.25:
            return True
            
        return False
    
    def _calculate_heuristic_prob(self, start_pos, obj_pos, target_pos, balls, tid, is_bank=False):
        """
        通用的几何概率计算。
        :param start_pos: 发起球位置 (通常是母球)
        :param obj_pos: 被击打球位置
        :param target_pos: 目标位置 (袋口 或 镜像袋口)
        """
        dist_1 = np.linalg.norm(obj_pos - start_pos)
        dist_2 = np.linalg.norm(target_pos - obj_pos)
        
        vec_1 = obj_pos - start_pos
        vec_2 = target_pos - obj_pos
        
        try:
            angle = pt.utils.angle(vec_1, vec_2)
            angle_deg = math.degrees(angle)
        except: 
            angle_deg = 90
            
        if angle_deg >= 85: return 0.0
        
        # 基础概率模型：角度越小越好，距离越短越好
        prob = (1.0 - angle_deg/90.0) * (1.0 / (1.0 + 0.5 * (dist_1 + dist_2)))
        
        # 如果是翻袋，天然难度加倍，概率打折
        if is_bank:
            prob *= 0.6
            
        return prob

    def generate_candidates(self, balls, my_targets, table):
        """
        生成高潜力的击球参数候选列表。
        覆盖: 直线球 (Straight), 翻袋 (Bank), 反弹 (Kick), 简单组合 (Combo)
        """
        candidates = []
        cue_ball = balls.get('cue')
        if not cue_ball: return []
        cue_pos = cue_ball.state.rvw[0]
        
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids: target_ids = ['8']

        # [修改] 1. 智能开球检测 (Triangle Detection)
        # 不再依赖具体的坐标或仅仅依赖数量，而是检测球的形态
        if 'cue' in balls and self._is_triangle_formation(balls):
            self.logger.info("[CueCard] Detecting Break Shot scenario (Triangle Formation).")
            return [self._get_optimized_break_shot()]

        # 2. 目标球预筛选 (Heuristic Filtering)
        primary_targets = self._filter_targets_heuristically(balls, target_ids, cue_pos, table)

        for tid in primary_targets:
            obj_pos = balls[tid].state.rvw[0]
            
            # --- Type A: 直线球 (Straight Shots) ---
            for pid, pocket in table.pockets.items():
                p_pos = pocket.center
                
                # 计算鬼球点 (Ghost Ball)
                aim_phi, aim_dist = self._get_aim_params(cue_pos, obj_pos, p_pos)
                
                # 几何检测
                if not self._is_path_blocked(cue_pos, obj_pos, balls, exclude=[tid]):
                    h_prob = self._calculate_heuristic_prob(cue_pos, obj_pos, p_pos, balls, tid)
                    # [关键改进] 增加角度微调 (+/- 0.2度)，寻找最稳的进球点
                    # 很多时候几何中心并不是物理最佳点(因为有throw效应)
                    for angle_offset in np.linspace(-0.5, 0.5, 5): 
                        for v in [1.5, 3.0, 4.0, 5.5]: 
                            candidates.append({'V0': v, 'phi': (aim_phi + angle_offset) % 360, 'theta': 0, 'a': 0, 'b': 0, 'type': 'straight','h_prob':h_prob})

            # --- Type B: 翻袋球 (Bank Shots) - 简单单库 ---
            # 原理: 镜像袋口
            # 2. Bank Shots (新增)
            self._generate_bank_shots(cue_pos, obj_pos, table, tid, balls, candidates)
            
            # 3. Kick Shots (新增 - 仅当直球很少或为了解球时才大量生成，防止候选爆炸)
            # 这里简单策略：如果直球少于5个，就尝试 Kick
            if len(candidates) < 5:
                self._generate_kick_shots(cue_pos, obj_pos, table, tid, candidates)

            # 只有在直球很少或者为了寻找更多机会时才调用，避免候选过多
            if len(candidates) < 5: 
                self._generate_combination_shots(cue_pos, balls, table, my_targets, candidates)

        # 补充：如果没有生成足够的有效击球，加入随机扰动
        if len(candidates) == 0:
             candidates.append(self._random_action())
             
        # [核心优化] 不再随机截断！
        # 1. 优先按 h_prob (几何概率) 排序，优先保留“看起来好打”的球
        # 2. 如果 h_prob 相同，再考虑随机性
        random.shuffle(candidates) # 先打乱，保证同分数的随机性
        candidates.sort(key=lambda x: x.get('h_prob', 0), reverse=True)
        
        # 3. 保留 Top 40 进入 L1 模拟 (之前只有 20)
        # 配合 n_l1_sims 降低到 10，总计算量 (40 * 10 = 400) 略高于之前 (20 * 15 = 300)，但覆盖面翻倍
        return candidates[:40]

    def _get_aim_params(self, cue_pos, obj_pos, target_pos):
        """计算瞄准角度 (Ghost Ball Logic)"""
        vec_obj_target = np.array(target_pos) - np.array(obj_pos)
        dist = np.linalg.norm(vec_obj_target)
        if dist == 0: return 0, 0
        
        # 鬼球位置
        ghost_pos = np.array(obj_pos) - (vec_obj_target / dist) * (2 * self.ball_radius)
        
        vec_cue_ghost = ghost_pos - np.array(cue_pos)
        phi = math.degrees(math.atan2(vec_cue_ghost[1], vec_cue_ghost[0])) % 360
        cue_dist = np.linalg.norm(vec_cue_ghost)
        
        return phi, cue_dist

    def _is_path_blocked(self, start, end, balls, exclude=[]):
        """优化的路径阻挡检测"""
        vec = np.array(end) - np.array(start)
        dist = np.linalg.norm(vec)
        if dist == 0: return False
        unit = vec / dist
        
        # 建立一个 bounding box 快速排除不相关的球
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
        margin = self.ball_radius * 2.1
        
        for bid, ball in balls.items():
            if bid == 'cue' or bid in exclude or ball.state.s == 4: continue
            
            b_pos = ball.state.rvw[0]
            # Bounding Box 预筛选
            if not (min_x - margin < b_pos[0] < max_x + margin and 
                    min_y - margin < b_pos[1] < max_y + margin):
                continue

            # 点到直线距离
            vec_b = np.array(b_pos) - np.array(start)
            proj = np.dot(vec_b, unit)
            
            # 球必须在 start 和 end 之间
            if 0 < proj < dist:
                perp_dist = np.linalg.norm(vec_b - proj * unit)
                # 判定阈值：2倍半径表示球心距离，稍微留点余量防止擦边太极限
                if perp_dist < 1.98 * self.ball_radius:
                    return True
        return False

    def _quick_noiseless_check(self, action, balls, table, targets):
        """Step 2 Filter: 快速无噪模拟验证可行性"""
        shot = self._simulate_shot(balls, table, action, noise=False)
        if shot is None: return False
        # 检查是否进球且不犯规
        return self._is_turn_kept(shot, balls, targets)

    # =========================================================================
    # Phase 2 & 3: Level 1 Search & Clustering (一级搜索与聚类)
    # =========================================================================


    def level_1_search_and_cluster(self, candidates, balls, table, targets):
        """
        执行 CueCard 的核心一级搜索：
        1. 对每个候选动作执行 N 次带噪模拟。
        2. 收集所有结果状态 (Resulting States)。
        3. 对结果状态进行聚类，提取 Representative States。
        4. 计算初步得分。
        """
        scored_candidates = []

        for action in candidates:
            result_states = []
            cumulative_score = 0
            
            # 基础分：引入启发式概率加权 (Heuristic Probability Bias)
            # 即使模拟因为噪声没进，如果这球几何上很好，也给它一点底分，避免被错杀
            h_prob = action.get('h_prob', 0.0)
            heuristic_bonus = h_prob * 40.0
            
            # --- Noisy Simulations ---
            for _ in range(self.n_l1_sims):
                shot = self._simulate_shot(balls, table, action, noise=True)
                if shot is None:
                    cumulative_score += -500 # 模拟失败惩罚
                    continue
                
                # 分析结果
                turn_kept = self._is_turn_kept(shot, balls, targets)
                final_balls = shot.balls
                
                # 计算该状态的静态评分 (State Evaluation)
                if turn_kept:
                    # [优化] 提高进球奖励 (100 -> 160)，鼓励进攻
                    # 加上启发式bonus，好球的得分上限更高
                    state_score = 160.0 + self._evaluate_state_probability(final_balls, targets, table) + heuristic_bonus
                else:
                    # [修正] 必须检查白球是否还在，否则会导致逻辑反转
                    if 'cue' not in final_balls:
                         # 母球洗袋，给予极大惩罚，不计算对手分数(因为那是-500)
                        state_score = -1000.0
                    else:
                        # 正常交换球权，Minimax: 我的得分 = 基础罚分 - 对手的优势
                        opp_targets = self._get_opponent_targets(targets)
                        opp_score = self._evaluate_state_probability(final_balls, opp_targets, table)
                        state_score = -50 - opp_score 
                
                cumulative_score += state_score
                
                # 收集用于聚类的特征: [Cue_X, Cue_Y, Score]
                # CueCard 论文指出聚类基于球的位置和状态评分
                cue_final_pos = final_balls['cue'].state.rvw[0]
                feature = [cue_final_pos[0], cue_final_pos[1], state_score]
                result_states.append({
                    'feature': feature,
                    'balls': final_balls, # 保存完整状态用于L2
                    'score': state_score
                })

            avg_score = cumulative_score / self.n_l1_sims
            
            # --- State Clustering (The Innovation) ---
            # 如果模拟产生的结果差异很大，使用 K-Means 提取代表性状态
            representatives = []
            if len(result_states) >= self.n_clusters:
                features_matrix = np.array([r['feature'] for r in result_states])
                
                # 动态调整聚类数：取实际状态数和期望聚类数的最小值
                # 避免聚类数超过不同状态数而产生警告
                actual_n_clusters = min(self.n_clusters, len(result_states))
                
                # 抑制 KMeans 的 ConvergenceWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    kmeans = KMeans(n_clusters=actual_n_clusters, n_init='auto', random_state=42).fit(features_matrix)
                
                # 找到距离每个聚类中心最近的真实状态
                for center in kmeans.cluster_centers_:
                    # 简单欧氏距离找最近邻
                    dists = np.linalg.norm(features_matrix - center, axis=1)
                    idx = np.argmin(dists)
                    representatives.append(result_states[idx])
            else:
                representatives = result_states # 样本太少直接全用

            scored_candidates.append({
                'action': action,
                'l1_score': avg_score,
                'representatives': representatives
            })
            
        # 按 L1 分数排序，只保留前 M 个进入 L2
        scored_candidates.sort(key=lambda x: x['l1_score'], reverse=True)
        return scored_candidates[:5] # 选 Top 5 进入精细搜索

    # =========================================================================
    # Phase 4: Level 2 Search (二级搜索)
    # =========================================================================

    def level_2_refinement(self, top_candidates, table, targets):
        """
        二级搜索：对代表性状态进行前瞻 (Lookahead)。
        如果剩下的球少于等于3个 (Endgame)，尝试模拟两步；否则只看当前局面评估。
        """
        best_action = None
        best_refined_score = -float('inf')
        
        # 判断是否进入 Endgame 模式
        remaining_balls = [b for b in targets if b != '8'] # 这里的 targets 可能包含 8
        is_endgame = (len(remaining_balls) <= 3)

        for item in top_candidates:
            action = item['action']
            reps = item['representatives']
            
            # 对该动作下的每一个代表性状态(可能产生的分支)进行评估
            rep_scores = []
            
            for rep in reps:
                state_balls = rep['balls']
                # 检查此状态游戏是否结束
                is_foul, turn_kept, game_res = self.analyze_shot_result_from_state(state_balls, targets) # 需简单封装
                
                if game_res == 'win':
                    rep_scores.append(10000) # 必胜路径
                    continue
                if game_res == 'lose':
                    rep_scores.append(-10000)
                    continue
                
                current_score = rep['score']
                
                # --- Lookahead Logic ---
                # 如果这个状态保住了球权，我们看看下一杆好不好打
                future_bonus = 0
                if turn_kept:
                    # 在这个新状态下，快速生成下一杆的最佳候选
                    # 为了速度，只生成直线球且只模拟几次
                    next_candidates = self.generate_candidates(state_balls, targets, table)
                    
                    if next_candidates:
                        # 快速取下一层最好的静态分 (不递归做L2，只看静态分)
                        # 我们假设下一杆能打出候选中的最佳分数
                        best_next_score = -float('inf')
                        for next_act in next_candidates[:3]: # 只看前3个
                             # 快速无噪模拟下一杆
                             next_shot = self._simulate_shot(state_balls, table, next_act, noise=False)
                             if next_shot:
                                 # 评估下一杆之后的局面
                                 s_score = self._evaluate_state_probability(next_shot.balls, targets, table)
                                 if s_score > best_next_score:
                                     best_next_score = s_score
                        
                        if best_next_score > -500:
                            future_bonus = best_next_score * 0.5 # 下一杆分数的折扣权重

                rep_scores.append(current_score + future_bonus)

            # 聚合分数：取平均值 (Average Case) 或 最小值 (Worst Case - 保守策略)
            # CueCard 倾向于平均值，但在关键球可能会保守
            final_score = np.mean(rep_scores) if rep_scores else -500
            
            self.logger.info(
                f"  > 动作 {action.get('type','未知')} | L1: {item['l1_score']:.1f} | L2: {final_score:.1f}"
            )

            if final_score > best_refined_score:
                best_refined_score = final_score
                best_action = action

        return best_action

    def analyze_shot_result_from_state(self, balls, targets):
        """辅助函数：仅基于球的状态判断游戏结果 (用于 L2 这里的静态分析)"""
        # 这是一个简化的 helper，因为 analyze_shot_result 需要 shot event
        # 这里只看球在哪里
        if 'cue' not in balls or balls['cue'].state.s == 4:
            return True, False, 'lose' # 假设白球进了就是输 (L2悲观估计)
        
        # [修复] 静态评估也需要正确的黑8检测
        eight_ball = balls.get('8')
        eight_pocketed = eight_ball is not None and eight_ball.state.s == 4

        if eight_pocketed:
            # 如果黑8没了，且 targets 只有8，则赢
            if len(targets) == 1 and targets[0] == '8': return False, True, 'win'
            else: return True, False, 'lose'
            
        return False, True, None # 默认还在继续

    # =========================================================================
    # Scoring Function (核心评估函数)
    # =========================================================================

    def _evaluate_state_probability(self, balls, targets, table):
        """
        CueCard 核心评估公式: Score = 1.0*p0 + 0.33*p1 + 0.15*p2 ...
        其中 pi 是打进第 i 个最容易球的概率。
        """
        if 'cue' not in balls: return -500 # 白球进袋

        # [修复] 检查黑8进袋状态时，要看 state.s 而不是 key是否存在
        # 如果8号球已经进袋 (s==4)
        eight_ball = balls.get('8')
        eight_pocketed = eight_ball is not None and eight_ball.state.s == 4
        
        if eight_pocketed:
            if '8' in targets: return 1000 # 我方合法打进，赢了
            else: return -1000 # 误进黑8，输了

        probs = []
        cue_pos = balls['cue'].state.rvw[0]
        
        valid_targets = [t for t in targets if t in balls]
        
        for tid in valid_targets:
            ball = balls[tid]
            b_pos = ball.state.rvw[0]
            
            # 对每个球，找最容易进的袋口计算概率
            best_prob = 0.0
            for pocket in table.pockets.values():
                p_pos = pocket.center
                
                # 1. 距离因子
                dist_cb = np.linalg.norm(b_pos - cue_pos)
                dist_bp = np.linalg.norm(p_pos - b_pos)
                if dist_bp == 0: continue
                
                # 2. 角度因子 (Cut Angle)
                vec_cb = b_pos - cue_pos
                vec_bp = p_pos - b_pos
                
                # 简化的切球角度计算
                try:
                    angle = pt.utils.angle(vec_cb, vec_bp) # 0 is straight
                    angle_deg = math.degrees(angle)
                except:
                    angle_deg = 90

                # 3. 概率估算模型 (Heuristic Probability)
                # 角度越大概率越低，距离越远概率越低
                # P = (1 - angle/90) * (1 / (1 + dist))
                if angle_deg >= 85:
                    prob = 0
                else:
                    prob = (1.0 - angle_deg/90.0) * (1.0 / (1.0 + 0.5 * (dist_cb + dist_bp)))
                
                # 4. 阻挡检测 (Penalty)
                if self._is_path_blocked(cue_pos, b_pos, balls, exclude=[tid]):
                    prob *= 0.1 # 即使阻挡也可能解球，但概率极低
                
                if prob > best_prob:
                    best_prob = prob
            
            probs.append(best_prob)
        
        # 排序: 最容易的球 p0, 第二容易 p1 ...
        probs.sort(reverse=True)
        
        # 加权求和
        weights = [1.0, 0.33, 0.15, 0.07, 0.03] # CueCard 权重递减
        score = 0
        for i, p in enumerate(probs):
            if i < len(weights):
                score += weights[i] * p * 100 # 放大分数
            else:
                break
                
        # 额外奖励：如果这是黑8且概率高
        if len(targets) == 1 and targets[0] == '8':
            if probs and probs[0] > 0.5:
                score += 500
                
        return score

    # =========================================================================
    # Helpers & Infrastructure
    # =========================================================================

    def decision(self, balls, my_targets, table):
        """主入口 (已集成 Safety Fallback)"""
        if balls is None: return self._random_action()
        self.logger.info(f"[CueCard] 思考中... 目标球: {my_targets}")


        # [核心修复] 修正目标球列表
        # 如果自己的球都进袋了(my_targets里的球都不在balls里)，则目标变为黑8
        # 注意: my_targets 只是一个ID列表，我们需要检查这些ID是否还在台面上
        remaining_group = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        
        if not remaining_group and '8' in balls and balls['8'].state.s != 4:
            self.logger.info("[CueCard] 组球已清，锁定黑8为合法目标。")
            actual_targets = ['8']
        else:
            actual_targets = my_targets
        
        # 1. Generate
        candidates = self.generate_candidates(balls, actual_targets, table)
        
        # 如果生成器连一个候选都生不出来（极罕见），直接防守
        if not candidates: 
            self.logger.warning("[CueCard] 没有生成候选动作，切换到安全模式。")
            return self._play_safety(balls, actual_targets, table)
        
        # 2. L1 Search + Cluster
        top_candidates = self.level_1_search_and_cluster(candidates, balls, table, actual_targets)
        
        # 3. L2 Refinement
        best_action = self.level_2_refinement(top_candidates, table, actual_targets)
        
        # ================= NEW: Safety Fallback Check =================
        # 我们需要判断 best_action 是否足够好。
        
        # 简便方法：我们修改一下 level_2_refinement 让它不仅仅返回 action，或者我们在外面重新评估一下
        # 但为了不修改太多原有结构，我们在这里做一个简单的阈值判断。
        # 如果 L1 的最高分都很低，且 L2 也没有明显提升，说明这球很难进。
        
        # 让我们遍历一下 top_candidates 看看最高的 L1 分数是多少
        # (CueCard 评分系统中，>0 通常意味着有进球概率，>50 是稳进)
        max_l1_score = max([c['l1_score'] for c in top_candidates]) if top_candidates else -999

        # 阈值设定: 
        # 如果最高分 < 20 (说明进球概率很低，或者只有很难的球)，考虑防守
        # 并且我们要比较 防守分数 vs 进攻分数 (这里简化处理：如果进攻分太低直接防守)
        if best_action is None or max_l1_score < -15.0:
            self.logger.info(f"[CueCard] 进攻前景不佳 (分数: {max_l1_score:.1f})，切换到安全模式。")
            return self._play_safety(balls, actual_targets, table)
        # ==============================================================
            
        return best_action

    def _simulate_shot(self, balls, table, action, noise=True):
        """物理模拟封装"""
        sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            V0, phi, theta = action['V0'], action['phi'], action['theta']
            a, b = action['a'], action['b']
            
            if noise:
                V0 = max(0.1, V0 + np.random.normal(0, self.noise_std['V0']))
                phi = (phi + np.random.normal(0, self.noise_std['phi'])) % 360
                # 其他参数噪声省略以节省计算...
            
            cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def analyze_shot_result(self, shot, last_balls, my_targets):
        """
        详细分析击球结果，返回：
        - is_foul (bool): 是否犯规
        - turn_kept (bool): 是否保住球权
        - game_over (str/None): 'win', 'lose' 或 None
        """
        # 1. 基础进球分析
        pocketed_ids = [bid for bid, b in shot.balls.items() 
                        if b.state.s == 4 and last_balls[bid].state.s != 4]
        
        cue_pocketed = 'cue' in pocketed_ids
        eight_pocketed = '8' in pocketed_ids
        own_pocketed = [bid for bid in pocketed_ids if bid in my_targets]
        
        # 2. 犯规判定 (First Contact)
        first_contact_id = None
        cushion_hit = False
        
        # 遍历事件寻找首个碰撞
        # 注意: pooltool 的 event 结构比较复杂，这里简化抓取关键信息
        valid_ball_ids = [str(i) for i in range(1, 16)]
        for e in shot.events:
            if not first_contact_id and e.event_type == pt.events.EventType.BALL_BALL:
                 # 假设 ids[0] 是主动球(通常cue), ids[1]是被动球
                 # 我们需要找到白球碰到的第一个球
                 ids = e.ids
                 if 'cue' in ids:
                     other = ids[1] if ids[0] == 'cue' else ids[0]
                     if other in valid_ball_ids:
                         first_contact_id = other
            
            if e.event_type in (
                pt.events.EventType.BALL_LINEAR_CUSHION,
                pt.events.EventType.BALL_CIRCULAR_CUSHION,
            ):
                cushion_hit = True

        # 判定 A: 母球洗袋
        if cue_pocketed:
            if eight_pocketed: return True, False, 'lose' # 白球+黑8 = 输
            return True, False, None # 犯规，交换球权

        # 判定 B: 黑8进袋
        if eight_pocketed:
            # 只有当目标仅仅是黑8（清台后）且合法击中时才算赢
            # 注意：my_targets 在 decision 中已经被修正为 ['8']，所以这里判断很简单
            if len(my_targets) == 1 and my_targets[0] == '8':
                if first_contact_id == '8': return False, True, 'win'
                else: return True, False, 'lose' # 没碰到8直接进了，或者先碰别的球（极少见）
            else:
                return True, False, 'lose' # 还没清台就打进8 = 输

        # 判定 C: 首球犯规
        is_open_table = False # 简化：假设已分色
        foul = False
        
        if first_contact_id is None:
            foul = True # 空杆
        else:
            # 必须先碰到自己的球 (如果是8号球阶段，必须先碰8)
            legal_contacts = my_targets
            if first_contact_id not in legal_contacts:
                foul = True
        
        if foul: return True, False, None

        # 判定 D: 吃库 (简化版：如果没有球进袋，且没有吃库 -> 犯规)
        if not pocketed_ids and not cushion_hit:
            # 严格规则其实更复杂(碰球后必须有球吃库)，这里简化
            return True, False, None 

        # 3. 球权判定
        # 合法进球 -> 保留球权
        if own_pocketed:
            return False, True, None
        
        # 没进球 -> 失去球权
        return False, False, None

    # 更新 _is_turn_kept 的调用逻辑
    def _is_turn_kept(self, shot, balls_before, my_targets):
        is_foul, turn_kept, game_res = self.analyze_shot_result(shot, balls_before, my_targets)
        return turn_kept

    def _get_opponent_targets(self, my_targets):
        """推断对手目标球"""
        all_solids = [str(i) for i in range(1, 8)]
        all_stripes = [str(i) for i in range(9, 16)]
        if set(my_targets).intersection(all_solids):
            return all_stripes
        return all_solids

    def _get_optimized_break_shot(self):
        """CueCard 的预设开球参数"""
        return {'V0': 6.5, 'phi': 0.0, 'theta': 0, 'a': 0, 'b': 0.2} # 稍微加塞防止白球死贴

    def _random_action(self):
        return {
            'V0': random.uniform(1.0, 5.0),
            'phi': random.uniform(0, 360),
            'theta': 0, 'a': 0, 'b': 0 ,'h_prob':0.0,'type':'random'
        }