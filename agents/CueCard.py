import numpy as np
import pooltool as pt
import copy
import random
import math
import logging
import concurrent.futures
from functools import partial
import os
import time
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("[Warning] cma库未安装，将使用传统候选生成。安装: pip install cma")

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

# =========================================================================
# Standalone Helper Functions (for Multiprocessing)
# =========================================================================

# 进程内计数器，用于同一进程内不同调用的区分
_worker_call_counter = 0

def _get_unique_seed():
    """
    生成唯一的随机种子，避免多进程/多次调用时种子碰撞。
    
    组合因素：
    1. time.time_ns(): 纳秒级时间戳
    2. os.getpid(): 进程ID（区分不同worker进程）
    3. 进程内计数器: 同一进程内不同调用的区分
    4. os.urandom(4): 系统级随机源（关键！每次调用都不同）
    
    关键保证：
    - os.urandom() 是系统级真随机源，每次调用返回不同值
    - 即使两个进程在同一纳秒启动，os.urandom() 也会给出不同结果
    """
    global _worker_call_counter
    _worker_call_counter += 1
    
    try:
        ts = time.time_ns()
    except AttributeError:
        ts = int(time.time() * 1e9)
    
    # os.urandom 是真随机源，这是防止碰撞的核心
    # 即使其他因素完全相同，os.urandom 也会不同
    random_bytes = os.urandom(4)
    random_int = int.from_bytes(random_bytes, 'little')
    
    # 组合所有因素（os.urandom 已经足够，其他是额外保险）
    seed = (ts ^ (os.getpid() << 20) ^ (_worker_call_counter << 10) ^ random_int) % (2**32)
    
    return seed

def _warmup_task(x):
    """进程池预热任务（必须是模块级别函数才能被pickle）"""
    return x * 2

def is_path_blocked_standalone(start, end, balls, ball_radius, exclude=[]):
    """优化的路径阻挡检测 (Standalone)"""
    vec = np.array(end) - np.array(start)
    dist = np.linalg.norm(vec)
    if dist == 0: return False
    unit = vec / dist
    
    min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
    min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
    margin = ball_radius * 2.1
    
    for bid, ball in balls.items():
        if bid == 'cue' or bid in exclude or ball.state.s == 4: continue
        
        b_pos = ball.state.rvw[0]
        if not (min_x - margin < b_pos[0] < max_x + margin and 
                min_y - margin < b_pos[1] < max_y + margin):
            continue

        vec_b = np.array(b_pos) - np.array(start)
        proj = np.dot(vec_b, unit)
        
        if 0 < proj < dist:
            perp_dist = np.linalg.norm(vec_b - proj * unit)
            if perp_dist < 1.98 * ball_radius:
                return True
    return False

def distance_to_nearest_pocket_standalone(ball_pos, table):
    """计算球到最近袋口的距离 (Standalone)"""
    min_dist = float('inf')
    for pocket in table.pockets.values():
        pocket_pos = pocket.center
        dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(pocket_pos[:2]))
        min_dist = min(min_dist, dist)
    return min_dist

def analyze_shot_result_standalone(shot, last_balls, my_targets):
    """详细分析击球结果 (Standalone)"""
    pocketed_ids = [bid for bid, b in shot.balls.items() 
                    if b.state.s == 4 and last_balls[bid].state.s != 4]
    
    cue_pocketed = 'cue' in pocketed_ids
    eight_pocketed = '8' in pocketed_ids
    own_pocketed = [bid for bid in pocketed_ids if bid in my_targets]
    
    first_contact_id = None
    cue_hit_cushion = False
    target_hit_cushion = False
    
    valid_ball_ids = [str(i) for i in range(1, 16)]
    for e in shot.events:
        if not first_contact_id and e.event_type == pt.events.EventType.BALL_BALL:
             ids = e.ids
             if 'cue' in ids:
                 other = ids[1] if ids[0] == 'cue' else ids[0]
                 if other in valid_ball_ids:
                     first_contact_id = other
        
        if e.event_type in (
            pt.events.EventType.BALL_LINEAR_CUSHION,
            pt.events.EventType.BALL_CIRCULAR_CUSHION,
        ):
            ids = getattr(e, 'ids', ())
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_id is not None and first_contact_id in ids:
                target_hit_cushion = True

    if cue_pocketed:
        if eight_pocketed: return True, False, 'lose' 
        return True, False, None 

    if eight_pocketed:
        if len(my_targets) == 1 and my_targets[0] == '8':
            if first_contact_id == '8': return False, True, 'win'
            else: return True, False, None 
        else:
            return True, False, 'lose' 

    if first_contact_id is None:
        return True, False, None  
    else:
        legal_contacts = my_targets
        if first_contact_id not in legal_contacts:
            return True, False, None  
    
    if (not pocketed_ids) and (not cue_hit_cushion) and (not target_hit_cushion):
        return True, False, None

    if own_pocketed:
        return False, True, None
    
    return False, False, None

def is_turn_kept_standalone(shot, balls_before, my_targets):
    is_foul, turn_kept, game_res = analyze_shot_result_standalone(shot, balls_before, my_targets)
    return turn_kept

def _calculate_heuristic_prob_standalone(start_pos, obj_pos, target_pos, balls, tid, ball_radius, is_bank=False):
    """
    Copy of _calculate_heuristic_prob logic for standalone use
    """
    dist_1 = np.linalg.norm(obj_pos - start_pos)
    dist_2 = np.linalg.norm(target_pos - obj_pos)
    total_dist = dist_1 + dist_2
    
    vec_1 = obj_pos - start_pos
    vec_2 = target_pos - obj_pos
    
    try:
        angle = pt.utils.angle(vec_1, vec_2)
        angle_deg = math.degrees(angle)
    except: 
        angle_deg = 90
        
    if angle_deg >= 80: return 0.0
    
    angle_factor = math.cos(math.radians(angle_deg)) 
    angle_factor = max(0.0, angle_factor)
    
    if angle_deg < 30:
        angle_factor = 1.0 - (angle_deg / 150.0)
    
    dist_factor = 1.0 / (1.0 + 0.1 * total_dist) 
    
    prob = angle_factor * dist_factor
    
    if is_bank:
        prob *= 0.55
        
    return prob

def evaluate_state_probability_standalone(balls, targets, table, ball_radius):
    """CueCard 核心评估公式 (Standalone, logic from CueCard.py)"""
    eight_ball = balls.get('8')
    eight_pocketed = eight_ball is not None and eight_ball.state.s == 4
    
    if eight_pocketed:
        if '8' in targets and 'cue' in balls and balls['cue'].state.s != 4: return 5000 
        else: return -10000 

    if 'cue' not in balls: return -1000 

    probs = []
    cue_pos = balls['cue'].state.rvw[0]
    
    valid_targets = [t for t in targets if t in balls]
    if not valid_targets and '8' in balls:
        valid_targets = ['8']
    
    for tid in valid_targets:
        ball = balls[tid]
        b_pos = ball.state.rvw[0]
        
        best_prob = 0.0
        for pocket in table.pockets.values():
            p_pos = pocket.center
            prob = _calculate_heuristic_prob_standalone(cue_pos, b_pos, p_pos, balls, tid, ball_radius)
            
            if is_path_blocked_standalone(cue_pos, b_pos, balls, ball_radius, exclude=[tid]):
                prob *= 0.1 
            
            if prob > best_prob:
                best_prob = prob
        
        probs.append(best_prob)
    
    probs.sort(reverse=True)
    
    weights = [1.0, 0.33, 0.15, 0.07, 0.03]
    score = 0
    for i, p in enumerate(probs):
        if i < len(weights):
            score += weights[i] * p * 500 
        else:
            break
            
    if len(targets) == 1 and targets[0] == '8':
        if probs and probs[0] > 0.8:
            score += 1000
    
    if probs and probs[0] < 0.3:
        table_w, table_l = table.w, table.l
        is_rail = (abs(cue_pos[0]) > table_w/2 - 0.1) or (abs(cue_pos[1]) > table_l/2 - 0.1)
        if is_rail: score += 20.0
    
    cue_dist_to_pocket = distance_to_nearest_pocket_standalone(cue_pos, table)
    if cue_dist_to_pocket < 0.1:
        danger_penalty = 500 * (0.1 - cue_dist_to_pocket) 
        score -= danger_penalty
    elif cue_dist_to_pocket > 0.2:
        score += min(50, cue_dist_to_pocket * 100)
    
    if '8' in balls and balls['8'].state.s != 4:
        is_targeting_eight = (len(targets) == 1 and targets[0] == '8')
        if not is_targeting_eight:
            eight_pos = balls['8'].state.rvw[0]
            eight_dist = distance_to_nearest_pocket_standalone(eight_pos, table)
            
            if eight_dist < 0.12:
                score -= 60 * (0.12 - eight_dist) / 0.12
            
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                cue_to_pocket = np.linalg.norm(np.array(cue_pos[:2]) - np.array(pocket_pos[:2]))
                eight_to_pocket = np.linalg.norm(np.array(eight_pos[:2]) - np.array(pocket_pos[:2]))
                if cue_to_pocket < 0.2 and eight_to_pocket < 0.2:
                    score -= 100
                    break
    return score

def simulate_shot_standalone(balls, table, action, noise_std, ball_radius, noise=True):
    """物理模拟封装 (Standalone)"""
    sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
    sim_table = copy.deepcopy(table)
    cue = pt.Cue(cue_ball_id="cue")
    shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
    
    try:
        V0, phi, theta = action['V0'], action['phi'], action['theta']
        a, b = action['a'], action['b']
        
        if noise:
            V0 = max(0.1, V0 + np.random.normal(0, noise_std['V0']))
            phi = (phi + np.random.normal(0, noise_std['phi'])) % 360
            a = a + np.random.normal(0, noise_std['a'])
            b = b + np.random.normal(0, noise_std['b'])
        
        if a**2 + b**2 > 1.0:
            norm = np.sqrt(a**2 + b**2)
            a = a / norm * 0.99
            b = b / norm * 0.99
        
        # poolenv.py 边界裁剪
        V0 = float(np.clip(V0, 0.5, 8.0))
        theta = float(np.clip(theta, 0.0, 90.0))
        a = float(np.clip(a, -0.5, 0.5))
        b = float(np.clip(b, -0.5, 0.5))
        
        cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
        pt.simulate(shot, inplace=True)
        return shot
    except Exception:
        return None

def worker_l1_search(action, balls, table, targets, n_sims, noise_std, ball_radius):
    """L1 搜索工作函数 (Standalone)"""
    # [RNG Fix] 使用唯一种子避免多进程/多次调用时的重复模拟
    np.random.seed(_get_unique_seed())
    
    cumulative_score = 0
    h_prob = action.get('h_prob', 0.0)
    heuristic_bonus = h_prob * 100.0
    
    kept_cnt = 0
    for sim_idx in range(n_sims):
        shot = simulate_shot_standalone(balls, table, action, noise_std, ball_radius, noise=True)
        if shot is None:
            cumulative_score += -500 
            remaining = n_sims - sim_idx - 1
            best_possible_keep_rate = (kept_cnt + remaining) / float(n_sims)
            if best_possible_keep_rate < 0.70:
                cumulative_score += -1000 * remaining
                break
            continue
        
        is_foul, turn_kept, game_res = analyze_shot_result_standalone(shot, balls, targets)
        final_balls = shot.balls
        
        if turn_kept:
            kept_cnt += 1
        
        if game_res == 'win':
            state_score = 2000.0
        elif game_res == 'lose':
            state_score = -8000.0
        elif is_foul:
            state_score = -500.0
        elif turn_kept:
            state_score = 300.0 + evaluate_state_probability_standalone(final_balls, targets, table, ball_radius) + heuristic_bonus
        else:
            state_score = -100.0
        
        cumulative_score += state_score

        remaining = n_sims - sim_idx - 1
        best_possible_keep_rate = (kept_cnt + remaining) / float(n_sims)
        if best_possible_keep_rate < 0.70:
            cumulative_score += -1000 * remaining
            break
        
    avg_score = cumulative_score / n_sims
    return {
        'action': action,
        'l1_score': avg_score
    }

def worker_play_safety(action, balls, table, my_targets, opp_targets, noise_std, ball_radius):
    """防守评估工作函数 (Standalone)"""
    np.random.seed(_get_unique_seed())
    
    shot = simulate_shot_standalone(balls, table, action, noise_std, ball_radius, noise=False)
    if shot:
        is_foul, _, _ = analyze_shot_result_standalone(shot, balls, my_targets)
        if is_foul:
            return None 

        if 'cue' not in shot.balls: return None

        final_cue_pos = shot.balls['cue'].state.rvw[0]
        dists = []
        blocked_count = 0
        
        opp_balls_pos = []
        for opp_id in opp_targets:
            if opp_id in shot.balls:
                op_pos = shot.balls[opp_id].state.rvw[0]
                opp_balls_pos.append(op_pos)
                dists.append(np.linalg.norm(final_cue_pos - op_pos))
        
        if not dists: 
            score = 0
        else:
            min_dist = min(dists)
            score = min_dist * 10.0
        
        table_w, table_l = shot.table.w, shot.table.l
        is_rail = (abs(final_cue_pos[0]) > table_w/2 - 0.06) or \
                  (abs(final_cue_pos[1]) > table_l/2 - 0.06)
        if is_rail: score += 30.0
        
        for op_pos in opp_balls_pos:
            if is_path_blocked_standalone(final_cue_pos, op_pos, shot.balls, ball_radius, exclude=[]):
                blocked_count += 1
        
        score += blocked_count * 50.0
        return {'action': action, 'score': score}
        
    return None

def worker_cma_eval(x, init_phi, initial_action_type, balls, table, targets, k_sims, noise_std, ball_radius):
    """CMA-ES 评估工作函数 (Standalone)"""
    np.random.seed(_get_unique_seed())

    a_val, b_val = x[2], x[3]
    if a_val**2 + b_val**2 > 1.0:
        norm = np.sqrt(a_val**2 + b_val**2)
        a_val = a_val / norm * 0.99
        b_val = b_val / norm * 0.99

    # poolenv.py 边界裁剪（与主进程逻辑保持一致）
    a_val = float(np.clip(a_val, -0.5, 0.5))
    b_val = float(np.clip(b_val, -0.5, 0.5))
    
    # poolenv.py 边界裁剪
    V0_val = float(np.clip(x[0], 0.5, 8.0))
    theta_val = 0.0 
    
    # x[1] 作为 delta_phi（局部优化），避免周期变量/尺度不一致问题
    phi_val = (float(init_phi) + float(x[1])) % 360.0
    act = {'V0': V0_val, 'phi': phi_val, 'theta': 0.0, 'a': a_val, 'b': b_val, 'type': initial_action_type}
    
    keep_cnt = 0
    foul_cnt = 0
    win_cnt = 0
    lose_cnt = 0
    state_scores = []
    
    is_shooting_8 = (len(targets) == 1 and targets[0] == '8')

    for _ in range(k_sims):
        shot = simulate_shot_standalone(balls, table, act, noise_std, ball_radius, noise=True)
        if shot is None:
            foul_cnt += 1
            state_scores.append(-1000.0)
            continue

        final_balls = shot.balls
        state_score = evaluate_state_probability_standalone(final_balls, targets, table, ball_radius)
        state_scores.append(state_score)

        is_foul, turn_kept, game_res = analyze_shot_result_standalone(shot, balls, targets)

        if is_shooting_8:
            if game_res == 'win':
                win_cnt += 1
            elif game_res == 'lose':
                lose_cnt += 1
            if is_foul:
                foul_cnt += 1
        else:
            if turn_kept:
                keep_cnt += 1
            if is_foul:
                foul_cnt += 1

    keep_rate = keep_cnt / float(k_sims)
    foul_rate = foul_cnt / float(k_sims)
    avg_state = float(np.mean(state_scores)) if state_scores else -1000.0

    if is_shooting_8:
        win_rate = win_cnt / float(k_sims)
        lose_rate = lose_cnt / float(k_sims)
        score = 20000.0 * win_rate - 50000.0 * lose_rate - 5000.0 * foul_rate + avg_state
        stats = {
            'k': k_sims,
            'win_rate': win_rate,
            'lose_rate': lose_rate,
            'foul_rate': foul_rate,
            'avg_state': avg_state,
        }
    else:
        # Match CueCard.py scoring
        score = (
            1200.0 * keep_rate
            - 4800.0 * foul_rate
            + 10.0 * avg_state * keep_rate
            - 400.0 * (1.0 - keep_rate)
        )
        stats = {
            'k': k_sims,
            'keep_rate': keep_rate,
            'foul_rate': foul_rate,
            'avg_state': avg_state,
        }
        
    return {
        'act': act,
        'score': score,
        'stats': stats
    }

def worker_cma_optimize_full(args):
    """
    完整的 CMA-ES 优化 worker (Standalone)
    """
    np.random.seed(_get_unique_seed())
    
    initial_action = args['initial_action']
    balls = args['balls']
    table = args['table']
    targets = args['targets']
    noise_std = args['noise_std']
    ball_radius = args['ball_radius']
    cma_maxiter = args['cma_maxiter']
    cma_popsize = args['cma_popsize']
    cma_eval_sims = args['cma_eval_sims']
    idx = args['idx']
    l1_score = args['l1_score']
    
    if not CMA_AVAILABLE:
        return {
            'idx': idx, 'action': initial_action, 'score': -1.0, 'stats': None, 'l1_score': l1_score
        }
    
    # Clip initial action
    V0 = float(np.clip(initial_action['V0'], 0.5, 8.0))
    a = float(np.clip(initial_action['a'], -0.5, 0.5))
    b = float(np.clip(initial_action['b'], -0.5, 0.5))

    init_phi = float(initial_action['phi'])
    phi_range_deg = 20.0

    # 优化变量改为 [V0, delta_phi, a, b]
    x0 = [V0, 0.0, a, b]
    lower_bounds = [0.5, -phi_range_deg, -0.5, -0.5]
    upper_bounds = [8.0, +phi_range_deg, 0.5, 0.5]
    
    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'popsize': max(2, int(cma_popsize)),
        'maxiter': max(1, int(cma_maxiter)),
        # 按维度设置初始步长（sigma0 不变仍为 0.25）
        'CMA_stds': [2.0, 20.0, 0.4, 0.4],
        'verbose': -9,
        'verb_log': 0,
    }
    
    try:
        es = cma.CMAEvolutionStrategy(x0, 0.25, opts)
    except:
        return {
            'idx': idx, 'action': initial_action, 'score': -1.0, 'stats': None, 'l1_score': l1_score
        }

    best_sol, best_score, best_stats = None, -float('inf'), None
    is_shooting_8 = (len(targets) == 1 and targets[0] == '8')
    
    if is_shooting_8:
        k_sims = 20
    else:
        k_sims = max(1, int(cma_eval_sims))

    # 早停：连续多代提升很小则停止
    stall_patience = 2
    stall_count = 0
    prev_gen_best = -float('inf')
    min_improvement = 200.0 if is_shooting_8 else 50.0

    while not es.stop():
        solutions = es.ask()
        fitness = []

        # 噪声下：先粗评估，再对 Top-N 复评
        if (not is_shooting_8) and k_sims >= 8:
            k_coarse = max(4, min(8, k_sims // 2))
            refine_top_n = 2
        else:
            k_coarse = k_sims
            refine_top_n = 0

        coarse_results = []  # [(idx, x, score, act, stats)]

        for sol_idx, x in enumerate(solutions):
            res = worker_cma_eval(
                x,
                init_phi=init_phi,
                initial_action_type=initial_action.get('type'),
                balls=balls, table=table, targets=targets,
                k_sims=k_coarse, noise_std=noise_std, ball_radius=ball_radius
            )
            # 重要：保留原始 x（包含 delta_phi），用于复评时保持与主进程一致，避免角度 wrap 引入歧义
            coarse_results.append((sol_idx, x, res['score'], res['act'], res['stats']))

        refined_scores = {}
        refined_stats = {}
        if refine_top_n > 0:
            coarse_sorted = sorted(coarse_results, key=lambda t: t[2], reverse=True)
            for sol_idx, x, _, _, _ in coarse_sorted[:refine_top_n]:
                res_full = worker_cma_eval(
                    x,
                    init_phi=init_phi,
                    initial_action_type=initial_action.get('type'),
                    balls=balls, table=table, targets=targets,
                    k_sims=k_sims, noise_std=noise_std, ball_radius=ball_radius
                )
                refined_scores[sol_idx] = res_full['score']
                refined_stats[sol_idx] = res_full['stats']

        gen_best = -float('inf')
        for sol_idx, _, score, act, stats in coarse_results:
            final_score = refined_scores.get(sol_idx, score)
            final_stats = refined_stats.get(sol_idx, stats)

            if final_score > best_score:
                best_score = final_score
                best_sol = act
                best_stats = final_stats
            if final_score > gen_best:
                gen_best = final_score

            fitness.append(-final_score)  # CMA minimized
        
        es.tell(solutions, fitness)

        # 早停：连续多代提升很小
        if gen_best > prev_gen_best:
            if (gen_best - prev_gen_best) < min_improvement:
                stall_count += 1
            else:
                stall_count = 0
            prev_gen_best = gen_best
        else:
            stall_count += 1
        if stall_count >= stall_patience:
            break

        if best_score > 5000:
            break

    if best_sol is None:
        return {
            'idx': idx, 'action': initial_action, 'score': -1.0, 'stats': None, 'l1_score': l1_score
        }
    
    best_sol['phi'] %= 360
    return {
        'idx': idx,
        'action': best_sol,
        'score': best_score,
        'stats': best_stats,
        'l1_score': l1_score
    }


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

    def __init__(self, use_cma=True):
        super().__init__()
        self.logger = _get_cuecard_logger()
        
        # --- 核心参数配置 (针对 8进程 优化) ---
        
        # 1. 第一层搜索 (L1 Search)
        self.n_l1_sims = 40          # L1 模拟次数 (32 -> 40, 提高评估稳定性)
        # 生成候选数量上限 (设为 8 的倍数，便于 8 进程并行负载均衡)
        self.num_candidates_generated = 48  
        # 进入 CMA 优化的候选数量 (8 个候选正好并行优化)
        self.num_candidates_cma = 8         

        # 2. CMA-ES 深度优化
        self.use_cma = use_cma and CMA_AVAILABLE
        self.cma_maxiter = 5                # CMA-ES迭代次数 (4 -> 5, 增加收敛性)
        self.cma_popsize = 8                # 每代种群大小 (10 -> 8, 与进程数对齐)
        # CMA评估时对同一动作做多次带噪模拟 (20 -> 24, 8的倍数)
        self.cma_eval_sims = 24
        
        # 3. 物理参数
        self.ball_radius = 0.028575
        self.pocket_radius = 0.05       # 估算值
        
        # 4. 噪声模型 (与 BasicAgentPro 对齐或根据环境设定)
        self.noise_std = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

        
        # 5. 进程池配置
        # 针对 8 核 CPU，开 8 个 worker 跑满算力
        self.max_workers = 1
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)

        if self.use_cma:
            self.logger.info("[CueCard] 代理初始化完成 (CMA-ES优化已启用, 8-Process Profile)")
        else:
            self.logger.info("[CueCard] 代理初始化完成 (CMA-ES优化未启用)")

        # 5. 进程池配置
        # 针对 8 进程，开 8 个 worker 跑满并行能力
        self.max_workers = 8
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        
        # 预热进程池，避免首次决策时的启动延迟（约3秒）
        self._warmup_executor()

    def _warmup_executor(self):
        """
        预热进程池：提交简单任务强制子进程启动。
        这样可以将约3秒的启动开销从首次决策转移到Agent初始化阶段。
        """
        start = time.time()
        
        # 提交简单任务到每个 worker（使用模块级函数，可被pickle）
        futures = [self._executor.submit(_warmup_task, i) for i in range(self.max_workers)]
        # 等待所有任务完成，确保子进程已完全启动
        for f in futures:
            f.result()
        
        elapsed = time.time() - start
        self.logger.info(f"[CueCard] 进程池预热完成 ({self.max_workers} workers, {elapsed:.2f}s)")

    def __del__(self):
        """析构时关闭进程池"""
        self.shutdown()

    def shutdown(self):
        """显式关闭进程池"""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

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
                # 添加候选 (增加力度档位覆盖面)
                for v in [3.0, 4.0, 5.0, 6.0]:
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
                
                # 生成候选 (Kick 通常需要较大力度，增加覆盖范围)
                for v in [3.5, 4.5, 5.5, 6.5]:
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
                    h_prob = prob_part2 * 0.4 # 组合球难度极大，直接打4折
                    
                    # 添加到候选 (组合球增加力度档位)
                    for v in [4.0, 5.0, 6.0]:
                        candidates.append({
                            'V0': v, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0,
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

        # 2. 评估防守效果 (并行化)
        best_safety_action = None
        best_safety_score = -float('inf')
        opp_targets = self._get_opponent_targets(my_targets)
        
        # 随机抽样 48 个候选进行评估 (8的倍数，适配8进程并行)
        random.shuffle(safety_candidates)
        candidates_to_eval = safety_candidates[:48]
        
        worker_func = partial(
            worker_play_safety,
            balls=balls, 
            table=table, 
            my_targets=my_targets, 
            opp_targets=opp_targets,
            noise_std=self.noise_std,
            ball_radius=self.ball_radius
        )
        
        results = []
        executor = self._executor
        if executor:
            # map returns iterator, cast to list to trigger computation if needed (though map computes eagerly usually)
            # But wait, map results are ordered. None means failed/foul.
            results = list(executor.map(worker_func, candidates_to_eval))
        else:
            results = [worker_func(c) for c in candidates_to_eval]
            
        # Parse results
        for res in results:
            if res is None: continue
            if res['score'] > best_safety_score:
                best_safety_score = res['score']
                best_safety_action = res['action']
        
        if best_safety_action:
            self.logger.info(f"[CueCard] 安全模式. 分数: {best_safety_score:.2f}")
            return best_safety_action
            
        # 实在没办法，只能随机，但力度要小
        return {'V0': 0.5, 'phi': random.uniform(0,360), 'theta':0, 'a':0, 'b':0}

    def _calculate_safety_score(self, shot, opp_targets):
        """
        计算防守分数：母球距离对手所有目标球越远越好。
        """

        if 'cue' not in shot.balls: return -1000.0 # 母球进袋，防守失败

        final_cue_pos = shot.balls['cue'].state.rvw[0]
        dists = []
        blocked_count = 0
        
        opp_balls_pos = []
        for opp_id in opp_targets:
            if opp_id in shot.balls:
                op_pos = shot.balls[opp_id].state.rvw[0]
                opp_balls_pos.append(op_pos)
                dists.append(np.linalg.norm(final_cue_pos - op_pos))
        
        if not dists: return 0
        
        # 策略：最大化"最近对手球的距离" (Max-Min Strategy)
        # 即：我不希望对手有任何简单的近球可打
        min_dist = min(dists)
        score = min_dist * 10.0 # 距离权重
        
        # 2. 贴库奖励
        table_w, table_l = shot.table.w, shot.table.l
        is_rail = (abs(final_cue_pos[0]) > table_w/2 - 0.06) or \
                  (abs(final_cue_pos[1]) > table_l/2 - 0.06)
        if is_rail: score += 30.0
        
        # 3. [新增] 斯诺克(做障碍)检测
        # 检查白球到对手每一个目标球之间是否有阻挡
        # 传入 shot.balls (未来的状态)
        for op_pos in opp_balls_pos:
            # 这里的 exclude 列表为空，因为任何球(包括我的球和8号)都可以作为障碍
            if self._is_path_blocked(final_cue_pos, op_pos, shot.balls, exclude=[]):
                blocked_count += 1
        
        # 每封锁一个对手的球，给予高额奖励
        score += blocked_count * 50.0
        return score
    

    def _filter_targets_heuristically(self, balls, target_ids, cue_pos, table):
        """
        智能预筛选(球,袋口)：对每个 (目标球, 袋口) 组合计算综合难度分 (角度 + 阻挡)，
        全部排序后只返回最容易打的几个组合，防止对极难的组合生成大量无效候选。

        返回: List[Tuple[target_id, pocket_id]]
        """
        scored_pairs = []
        
        for tid in target_ids:
            ball = balls[tid]
            obj_pos = ball.state.rvw[0]

            # 遍历所有袋口：为每个 (tid, pid) 计算难度
            for pid, pocket in table.pockets.items():
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
                
                # 角度评分：与 "之前打分"（_calculate_heuristic_prob / _evaluate_state_probability）保持一致
                # - angle>=85° 视为几乎不可能
                # - 小角度更宽容：cos 衰减 + <30° 的近似线性修正
                if angle_deg >= 80:
                    difficulty = 10000.0
                else:
                    angle_factor = math.cos(math.radians(angle_deg))
                    angle_factor = max(0.0, angle_factor)
                    if angle_deg < 30:
                        angle_factor = 1.0 - (angle_deg / 150.0)

                    # 2. 阻挡检测 (Blocking Penalty) - 关键！
                    # 检查 母球->鬼球 路径 (排除目标球自身 tid)
                    is_cue_blocked = self._is_path_blocked(cue_pos, ghost_pos, balls, exclude=[tid])
                    # 检查 目标球->袋口 路径
                    is_obj_blocked = self._is_path_blocked(obj_pos, p_pos, balls, exclude=[tid])

                    block_penalty = 0.0
                    if is_cue_blocked:
                        block_penalty += 5000.0
                    if is_obj_blocked:
                        block_penalty += 5000.0

                    # [niceAgent优化] 3. 黑8邻近惩罚 - 避免打黑8附近的球
                    eight_proximity_penalty = 0.0
                    if tid != '8' and '8' in balls and balls['8'].state.s != 4:
                        eight_pos = balls['8'].state.rvw[0]
                        dist_to_eight = np.linalg.norm(np.array(obj_pos[:2]) - np.array(eight_pos[:2]))
                        min_safe_distance = 0.3

                        if dist_to_eight < min_safe_distance:
                            proximity_ratio = (min_safe_distance - dist_to_eight) / min_safe_distance
                            eight_proximity_penalty = (proximity_ratio ** 2) * 3000.0

                    # 综合难度：删除距离因子，仅用角度 + 阻挡 + 黑8邻近
                    # angle_factor 越大越容易，因此难度用 (1-angle_factor) 表示
                    difficulty = (1.0 - angle_factor) * 1000.0 + block_penalty + eight_proximity_penalty

                scored_pairs.append((tid, pid, float(difficulty)))
            
        # 按难度排序 (分数越低越好)
        scored_pairs.sort(key=lambda x: x[2])

        # 只保留最容易打的 8 个 (球,袋口) 组合进入候选生成
        # 增加到 8 个以提高覆盖面，同时与 8 进程并行对齐
        return [(tid, pid) for tid, pid, _ in scored_pairs[:8]]
    
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
        
        # 4. 计算紧密度
        distances = np.linalg.norm(positions - centroid, axis=1)
        if np.max(distances) > 0.25: return False # 球没聚在一起
        
        # 5. [新增] 位置检测 (关键修复)
        # 标准台球桌长 L，原点在中心。脚点(Foot Spot)通常在 x=0, y=L/4 (或 -L/4)
        # 检查质心是否在纵轴的 1/4 处附近 (允许正负，适应不同半场)
        # 假设 table.l 约为 2.24m (7ft) - 2.84m (9ft)
        # 只要质心绝对值在 L/4 附近 (0.4m ~ 0.8m 范围内)，才认为是开球堆
        # 如果在台面中心 (0,0)，那肯定不是开球
        if abs(centroid[1]) < 0.3: return False 

        return True

    def _get_closest_object_ball(self, balls):
        """返回距离白球最近的台面目标球 (ball_id, distance)。

        说明：这里的“目标球”是指非白球、且未进袋的任意物球（1-15/8）。
        """
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return None, float('inf')

        cue_pos = cue_ball.state.rvw[0]
        best_id = None
        best_dist = float('inf')

        for bid, ball in balls.items():
            if bid == 'cue' or ball.state.s == 4:
                continue
            try:
                pos = ball.state.rvw[0]
            except Exception:
                continue
            d = float(np.linalg.norm(np.array(pos) - np.array(cue_pos)))
            if d < best_dist:
                best_dist = d
                best_id = bid
        return best_id, best_dist

    def _classify_ball_for_player(self, ball_id, my_targets):
        """把 ball_id 分类为 target / non-target / eight / unknown。"""
        if ball_id is None:
            return 'unknown'
        if ball_id == '8':
            return 'eight'
        if my_targets is None:
            return 'unknown'
        return 'target' if ball_id in my_targets else 'non-target'
    
    
    def _calculate_heuristic_prob(self, start_pos, obj_pos, target_pos, balls, tid, is_bank=False):
        """
        通用的几何概率计算。
        :param start_pos: 发起球位置 (通常是母球)
        :param obj_pos: 被击打球位置
        :param target_pos: 目标位置 (袋口 或 镜像袋口)
        """
        dist_1 = np.linalg.norm(obj_pos - start_pos)
        dist_2 = np.linalg.norm(target_pos - obj_pos)
        total_dist = dist_1 + dist_2
        
        vec_1 = obj_pos - start_pos
        vec_2 = target_pos - obj_pos
        
        try:
            angle = pt.utils.angle(vec_1, vec_2)
            angle_deg = math.degrees(angle)
        except: 
            angle_deg = 90
            
        if angle_deg >= 80: return 0.0
        
        # --- [修改点] 非线性角度惩罚 ---
        
        # 使用余弦衰减，对小角度容忍度高
        angle_factor = math.cos(math.radians(angle_deg)) 
        # 再次平滑，防止负数
        angle_factor = max(0.0, angle_factor)
        
        # 如果角度很小（<30度），认为是必进的 (接近 1.0)
        if angle_deg < 30:
            angle_factor = 1.0 - (angle_deg / 150.0) # 30度时还有 0.8
        
        # --- 距离惩罚 ---
        # 距离越远，精度越低。
        dist_factor = 1.0 / (1.0 + 0.1 * total_dist) # 降低距离权重的衰减系数 0.5 -> 0.3
        
        prob = angle_factor * dist_factor
        
        # 翻袋难度系数
        if is_bank:
            prob *= 0.55
            
        return prob

    def generate_candidates(self, balls, my_targets, table):
        """
        生成高潜力的击球参数候选列表。
        策略：
        1) 先对预筛选出的少量 (球,袋口) 组合生成直线球候选；
        2) 若直线球数量不足，再补充 bank/kick/combo；
        3) 仍为空时才使用随机动作兜底。
        """
        candidates = []
        cue_ball = balls.get('cue')
        if not cue_ball: return []
        cue_pos = cue_ball.state.rvw[0]
        
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids: target_ids = ['8']


        # 2. (球,袋口) 预筛选 (Heuristic Filtering)
        # 仅对最简单的 5 个 (tid, pid) 组合生成候选，控制候选规模
        primary_pairs = self._filter_targets_heuristically(balls, target_ids, cue_pos, table)

        for tid, pid in primary_pairs:
            if tid not in balls or pid not in table.pockets:
                continue

            obj_pos = balls[tid].state.rvw[0]
            p_pos = table.pockets[pid].center

            # 计算瞄准角度 (Ghost Ball)
            aim_phi, aim_dist = self._get_aim_params(cue_pos, obj_pos, p_pos)


            h_prob = self._calculate_heuristic_prob(cue_pos, obj_pos, p_pos, balls, tid)

            # 直球候选：根据距离自适应确定力度，并加小扰动
            # 经验：目标球->袋口更远需要更大力度；母球->目标球越远也需要更大力度以保持准度
            dist_cb = float(np.linalg.norm(np.array(obj_pos) - np.array(cue_pos)))
            dist_bp = float(np.linalg.norm(np.array(p_pos) - np.array(obj_pos)))
            total_dist = dist_cb + dist_bp

            # 基础力度（系数按常见 7ft/9ft 尺度做保守估计，可继续调参）
            v_base = 1.2 + 0.75 * dist_bp + 0.35 * dist_cb
            v_base = float(np.clip(v_base, 0.8, 8.0))

            # 力度扰动：围绕 v_base 取 4 档，平衡覆盖面与精度
            v_list = [float(np.clip(v_base + dv, 0.8, 8.0)) for dv in (-0.3, 0.0, 0.3, 0.6)]

            # 角度扰动：保守范围，避免精度损失
            # 远距离球更敏感，给更小的扰动；近距离球可以稍大
            phi_jitter = 0.25 / (1.0 + 0.5 * total_dist)
            phi_jitter = float(np.clip(phi_jitter, 0.08, 0.20))
            phi_list = [aim_phi - phi_jitter, aim_phi, aim_phi + phi_jitter]

            # 旋转参数：只在近距离球时启用（远距离容易放大误差）
            # a: 侧旋 (左-/右+), b: 顶旋(+)/缩杆(-)
            if total_dist < 1.2:  # 近距离球可以考虑走位
                spin_combos = [
                    (0.0, 0.0),      # 无旋转 (默认)
                    (0.0, 0.12),     # 轻微顶杆（跟进）
                    (0.0, -0.12),    # 轻微缩杆（定杆/回缩）
                ]
            else:  # 远距离球只用无旋转，保证精度
                spin_combos = [(0.0, 0.0)]

            for v in v_list:
                for opt_phi in phi_list:
                    for a_spin, b_spin in spin_combos:
                        candidates.append({
                            'V0': v, 'phi': opt_phi, 'theta': 0,
                            'a': a_spin, 'b': b_spin,
                            'type': 'straight', 'h_prob': h_prob,
                        })

        # 3) 始终生成 Bank/Kick/Combo 候选作为补充战术选择
        # 这些特殊球型在某些局面下可能是最优解
        selected_tids = []
        seen = set()
        for tid, _ in primary_pairs:
            if tid not in seen:
                seen.add(tid)
                selected_tids.append(tid)

        # 为前 3 个最容易打的目标球生成 Bank 和 Kick 候选
        for tid in selected_tids[:3]:
            if tid not in balls or balls[tid].state.s == 4:
                continue
            obj_pos = balls[tid].state.rvw[0]
            self._generate_bank_shots(cue_pos, obj_pos, table, tid, balls, candidates)
            self._generate_kick_shots(cue_pos, obj_pos, table, tid, candidates)

        # 组合球候选
        self._generate_combination_shots(cue_pos, balls, table, my_targets, candidates)

        self.logger.info(f"[候选生成] 共生成 {len(candidates)} 个几何候选")
        
        # 补充：如果没有生成足够的有效击球，加入随机扰动
        if len(candidates) == 0:
             candidates.append(self._random_action())
        
        # 1. 优先按 h_prob (几何概率) 排序，优先保留“看起来好打”的球
        # 2. 如果 h_prob 相同，再考虑随机性
        random.shuffle(candidates) # 先打乱，保证同分数的随机性
        candidates.sort(key=lambda x: x.get('h_prob', 0), reverse=True)
        
        
        # 直接保留 Top N 进入 L1 模拟
        # 注：稀疏过滤已移除，因为候选生成时的扰动步长已足够大，不会产生重复候选
        return candidates[:self.num_candidates_generated]

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
        return is_path_blocked_standalone(start, end, balls, self.ball_radius, exclude)


    # =========================================================================
    # Phase 2: Level 1 Search (一级搜索)
    # =========================================================================

    def level_1_search(self, candidates, balls, table, targets):
        """
        执行 CueCard 的一级搜索：
        1. 对每个候选动作执行 N 次带噪声的模拟。
        2. 聚合得到该动作的平均得分 (L1 score)。
        3. 按 L1 score 选出 TopK 进入后续（CMA）精修。
        """
        scored_candidates = []

        # Multiprocessing Execution
        worker_func = partial(worker_l1_search,
                              balls=balls, table=table, targets=targets,
                              n_sims=self.n_l1_sims, noise_std=self.noise_std, 
                              ball_radius=self.ball_radius)

        executor = self._executor
        # executor.map 保持顺序，且 results 会等待所有计算完成
        # 注意：这里我们假设 _executor 已经初始化
        if executor:
            results = list(executor.map(worker_func, candidates))
        else:
            # Fallback for safety
            self.logger.warning("[CueCard] 进程池未初始化，使用串行 L1 搜索")
            results = [worker_func(c) for c in candidates]
            
        # 处理结果
        # results 已经是包含 {'action': ..., 'l1_score': ...} 的列表
        scored_candidates = results
            
        # 按 L1 分数排序，只保留前 M 个进入 L2
        scored_candidates.sort(key=lambda x: x['l1_score'], reverse=True)
        return scored_candidates[:self.num_candidates_cma] # 选 Top K 进入精细搜索

    # =========================================================================
    # Phase 3: cma-ES Optimization (CMA-ES 优化)
    # =========================================================================

    def _cma_optimize_shot(self, initial_action, balls, table, targets):
        """CMA 优化函数 (方案B：用多次带噪模拟估计鲁棒性成功率)"""
        if not self.use_cma:
            return initial_action, -1.0, None

        # poolenv.py 会对动作做 clip，这里必须对齐边界，避免“仿真评估的动作”和“真实执行的动作”不一致
        def _clip_action_params(V0, theta, a, b):
            V0 = float(np.clip(V0, 0.5, 8.0))
            theta = float(np.clip(theta, 0.0, 90.0))
            a = float(np.clip(a, -0.5, 0.5))
            b = float(np.clip(b, -0.5, 0.5))
            return V0, theta, a, b
        
        # 初始参数与边界
        # 重要：phi 是周期变量且尺度(度)与其他维度不同。
        # 这里用“delta_phi”替代直接优化绝对 phi：
        #   x = [V0, delta_phi, a, b]
        #   phi = init_phi + delta_phi
        init_V0, _, init_a, init_b = _clip_action_params(
            initial_action['V0'], 0.0, initial_action['a'], initial_action['b']
        )
        init_phi = float(initial_action['phi'])
        phi_range_deg = 20.0
        x0 = [init_V0, 0.0, init_a, init_b]
        lower_bounds = [0.5, -phi_range_deg, -0.5, -0.5]
        upper_bounds = [8.0, +phi_range_deg, 0.5, 0.5]
        
        # CMA配置
        opts = {
            'bounds': [lower_bounds, upper_bounds],
            'popsize': max(2, int(self.cma_popsize)),
            'maxiter': max(1, int(self.cma_maxiter)),
            # 按维度设置初始步长（解决各参数尺度不一致导致的探索不充分/过度）
            # 注意：cma 中实际初始步长约为 sigma0 * CMA_stds。
            # 这里 sigma0 仍保持原值 0.25，不改你的其它超参。
            # 目标初始步长：V0≈0.5, delta_phi≈5deg, a/b≈0.1
            'CMA_stds': [2.0, 20.0, 0.4, 0.4],
            'verbose': -9,
            'verb_log': 0,
        }
        
        try:
            es = cma.CMAEvolutionStrategy(x0, 0.25, opts)
        except:
            return initial_action, -1.0, None

        best_sol, best_score, best_stats = None, -float('inf'), None
        is_shooting_8 = (len(targets) == 1 and targets[0] == '8')
        # 黑八阶段：提高评估次数，降低小样本高估胜率的偏差
        if is_shooting_8:
            k_sims = 20
        else:
            k_sims = max(1, int(getattr(self, 'cma_eval_sims', 1)))

        # 早停：若连续多代提升很小，则提前停止（噪声评估下可显著省时）
        stall_patience = 2
        stall_count = 0
        prev_gen_best = -float('inf')
        # 分数尺度较大，这里用一个相对保守的固定阈值
        min_improvement = 200.0 if is_shooting_8 else 50.0

        def _evaluate_action_with_noise(act, num_sims: int):
            """对同一动作做 num_sims 次带噪模拟，返回 (score, stats)"""
            keep_cnt = 0
            foul_cnt = 0
            win_cnt = 0
            lose_cnt = 0
            state_scores = []

            for _ in range(num_sims):
                shot = self._simulate_shot(balls, table, act, noise=True)
                if shot is None:
                    foul_cnt += 1
                    state_scores.append(-1000.0)
                    continue

                final_balls = shot.balls
                state_score = self._evaluate_state_probability(final_balls, targets, table)
                state_scores.append(state_score)

                is_foul, turn_kept, game_res = self.analyze_shot_result(shot, balls, targets)

                if is_shooting_8:
                    if game_res == 'win':
                        win_cnt += 1
                    elif game_res == 'lose':
                        lose_cnt += 1
                    if is_foul:
                        foul_cnt += 1
                else:
                    if turn_kept:
                        keep_cnt += 1
                    if is_foul:
                        foul_cnt += 1

            avg_state = float(np.mean(state_scores)) if state_scores else -1000.0

            if is_shooting_8:
                win_rate = win_cnt / float(num_sims)
                lose_rate = lose_cnt / float(num_sims)
                foul_rate = foul_cnt / float(num_sims)
                score = 20000.0 * win_rate - 50000.0 * lose_rate - 5000.0 * foul_rate + avg_state
                stats = {
                    'k': num_sims,
                    'win_rate': win_rate,
                    'lose_rate': lose_rate,
                    'foul_rate': foul_rate,
                    'avg_state': avg_state,
                }
            else:
                keep_rate = keep_cnt / float(num_sims)
                foul_rate = foul_cnt / float(num_sims)
                score = (
                    1200.0 * keep_rate
                    - 4800.0 * foul_rate
                    + 10.0 * avg_state * keep_rate
                    - 400.0 * (1.0 - keep_rate)
                )
                stats = {
                    'k': num_sims,
                    'keep_rate': keep_rate,
                    'foul_rate': foul_rate,
                    'avg_state': avg_state,
                }
            return score, stats

        while not es.stop():
            solutions = es.ask()
            fitness = []

            # 方案：噪声下先用较小 k 粗评估全部解，然后对 Top-N 做 full k 复评
            # 这样能减少“小样本高估”同时节省大量计算。
            if (not is_shooting_8) and k_sims >= 8:
                k_coarse = max(4, min(8, k_sims // 2))
                refine_top_n = 2
            else:
                k_coarse = k_sims
                refine_top_n = 0

            coarse_results = []  # [(idx, score, act, stats)]

            for sol_idx, x in enumerate(solutions):
                # [修复] 确保 a² + b² <= 1.0 (单位圆约束)
                a_val, b_val = x[2], x[3]
                if a_val**2 + b_val**2 > 1.0:
                    # 将 (a, b) 投影到单位圆上
                    norm = np.sqrt(a_val**2 + b_val**2)
                    a_val = a_val / norm * 0.99  # 0.99 留一点余量
                    b_val = b_val / norm * 0.99

                V0_val, theta_val, a_val, b_val = _clip_action_params(x[0], 0.0, a_val, b_val)
                # x[1] 是 delta_phi
                phi_val = (init_phi + float(x[1])) % 360.0
                act = {
                    'V0': V0_val,
                    'phi': phi_val,
                    'theta': theta_val,
                    'a': a_val,
                    'b': b_val,
                    'type': initial_action.get('type'),
                }

                score, stats = _evaluate_action_with_noise(act, k_coarse)
                coarse_results.append((sol_idx, score, act, stats))

            # 对 Top-N 解做 full k 复评（稳健）
            refined_scores = {}
            refined_stats = {}
            if refine_top_n > 0:
                coarse_results_sorted = sorted(coarse_results, key=lambda t: t[1], reverse=True)
                for sol_idx, _, act, _ in coarse_results_sorted[:refine_top_n]:
                    full_score, full_stats = _evaluate_action_with_noise(act, k_sims)
                    refined_scores[sol_idx] = full_score
                    refined_stats[sol_idx] = full_stats

            # 生成 fitness，并更新 best
            gen_best = -float('inf')
            for sol_idx, score, act, stats in coarse_results:
                final_score = refined_scores.get(sol_idx, score)
                final_stats = refined_stats.get(sol_idx, stats)

                if final_score > best_score:
                    best_score = final_score
                    best_sol = act
                    best_stats = final_stats

                if final_score > gen_best:
                    gen_best = final_score

                fitness.append(-final_score)  # CMA 求最小

            es.tell(solutions, fitness)

            # 早停：连续多代提升很小
            if gen_best > prev_gen_best:
                if (gen_best - prev_gen_best) < min_improvement:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_gen_best = gen_best
            else:
                stall_count += 1

            if stall_count >= stall_patience:
                break

            if best_score > 5000: break 

        if best_sol is None:
            return initial_action, -1.0, None
        best_sol['phi'] %= 360
        return best_sol, best_score, best_stats

    # =========================================================================
    # Scoring Function (核心评估函数)
    # =========================================================================

    def _evaluate_state_probability(self, balls, targets, table):
        """
        CueCard 核心评估公式: Score = 1.0*p0 + 0.33*p1 + 0.15*p2 ...
        其中 pi 是打进第 i 个最容易球的概率。
        """
        
        # 如果8号球已经进袋 (s==4)
        eight_ball = balls.get('8')
        eight_pocketed = eight_ball is not None and eight_ball.state.s == 4
        
        if eight_pocketed:
            if '8' in targets and 'cue' in balls and balls['cue'].state.s != 4: return 5000 # 我方合法打进，赢了
            else: return -10000 # 误进黑8，输了

        if 'cue' not in balls: return -1000 # 白球进袋

        probs = []
        cue_pos = balls['cue'].state.rvw[0]
        
        valid_targets = [t for t in targets if t in balls]

        # 如果没有目标球了(理论上应该只剩8)，但8没进，说明还在过渡期
        if not valid_targets and '8' in balls:
            valid_targets = ['8']
        
        for tid in valid_targets:
            ball = balls[tid]
            b_pos = ball.state.rvw[0]
            
            # 对每个球，找最容易进的袋口计算概率
            best_prob = 0.0
            for pocket in table.pockets.values():
                p_pos = pocket.center
                # 直接复用通用概率模型，避免重复计算流程
                prob = self._calculate_heuristic_prob(cue_pos, b_pos, p_pos, balls, tid)
                
                # 阻挡检测 (Penalty)
                if self._is_path_blocked(cue_pos, b_pos, balls, exclude=[tid]):
                    prob *= 0.1 # 即使阻挡也可能解球，但概率极低
                
                if prob > best_prob:
                    best_prob = prob
            
            probs.append(best_prob) # 记录该球的最佳进袋概率
        
        # 排序: 最容易的球 p0, 第二容易 p1 ...
        probs.sort(reverse=True)
        
        # 加权求和
        weights = [1.0, 0.33, 0.15, 0.07, 0.03] # CueCard 权重递减
        score = 0
        for i, p in enumerate(probs):
            if i < len(weights):
                score += weights[i] * p * 500 # 放大分数
            else:
                break
                
        # 额外奖励：如果这是黑8且概率高
        if len(targets) == 1 and targets[0] == '8':
            if probs and probs[0] > 0.8:
                score += 1000
        
        # 母球位置安全性奖励 (简单的斯诺克自我保护)
        # 如果这一杆打不进(最高概率很低)，希望白球停在库边
        if probs and probs[0] < 0.3:
            table_w, table_l = table.w, table.l
            is_rail = (abs(cue_pos[0]) > table_w/2 - 0.1) or (abs(cue_pos[1]) > table_l/2 - 0.1)
            if is_rail: score += 20.0
        
        # 密集奖励信号 - 母球危险位置惩罚
        cue_dist_to_pocket = self._distance_to_nearest_pocket(cue_pos, table)
        if cue_dist_to_pocket < 0.1:
            # 母球太接近袋口，按距离递增惩罚
            danger_penalty = 500 * (0.1 - cue_dist_to_pocket) 
            score -= danger_penalty
        elif cue_dist_to_pocket > 0.2:
            # 母球远离袋口，小额奖励
            score += min(50, cue_dist_to_pocket * 100)
        
        #  黑8安全距离监控（非打黑8阶段）
        if '8' in balls and balls['8'].state.s != 4:
            is_targeting_eight = (len(targets) == 1 and targets[0] == '8')
            if not is_targeting_eight:
                eight_pos = balls['8'].state.rvw[0]
                eight_dist = self._distance_to_nearest_pocket(eight_pos, table)
                
                # 黑8太接近袋口，大额惩罚
                if eight_dist < 0.12:
                    score -= 60 * (0.12 - eight_dist) / 0.12
                
                # 组合风险：母球和黑8同时接近同一袋口
                for pocket in table.pockets.values():
                    pocket_pos = pocket.center
                    cue_to_pocket = np.linalg.norm(np.array(cue_pos[:2]) - np.array(pocket_pos[:2]))
                    eight_to_pocket = np.linalg.norm(np.array(eight_pos[:2]) - np.array(pocket_pos[:2]))
                    
                    if cue_to_pocket < 0.2 and eight_to_pocket < 0.2:
                        score -= 100  # 双重风险大幅惩罚
                        break
                
        return score

    # =========================================================================
    # Helpers & Infrastructure
    # =========================================================================

    def decision(self, balls, my_targets, table):
        """主入口 (已集成 Safety Fallback)"""
        if balls is None: return self._random_action()
        self.logger.info(f"[CueCard] 思考中... 目标球: {my_targets}")

        # 开球阶段：直接使用预设开球参数，不进入后续 L1/CMA 判断
        if 'cue' in balls and self._is_triangle_formation(balls):
            closest_id, closest_dist = self._get_closest_object_ball(balls)
            closest_type = self._classify_ball_for_player(closest_id, my_targets)
            if closest_type == 'target':
                action = self._get_break_shot_closest_is_target()
            else:
                action = self._get_break_shot_closest_is_nontarget()

            self.logger.info(
                f"[CueCard] 开球阶段早返回: 最近球={closest_id} (类型={closest_type}, 距离={closest_dist:.3f}) -> {action.get('type')}"
            )
            return action


        # [核心修复] 修正目标球列表
        # 如果自己的球都进袋了(my_targets里的球都不在balls里)，则目标变为黑8
        # 注意: my_targets 只是一个ID列表，我们需要检查这些ID是否还在台面上
        remaining_group = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        
        if not remaining_group and '8' in balls and balls['8'].state.s != 4:
            self.logger.info("[CueCard] 组球已清，锁定黑8为合法目标。")
            actual_targets = ['8']
        else:
            actual_targets = my_targets
        
        is_shooting_8 = (actual_targets == ['8'])

        # 1. Generate
        candidates = self.generate_candidates(balls, actual_targets, table)
        
        # 如果生成器连一个候选都生不出来（极罕见），直接防守
        if not candidates: 
            self.logger.warning("[CueCard] 没有生成候选动作，切换到安全模式。")
            return self._play_safety(balls, actual_targets, table)
        
        # [调试] 输出候选类型分布（在L1之前）
        type_counts = {}
        for c in candidates:
            t = c.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        self.logger.info(f"[候选统计] 生成 {len(candidates)} 个候选, 类型: {type_counts}")
        
        # 2. L1 Search 
        top_candidates = self.level_1_search(candidates, balls, table, actual_targets)
        
        # 3. CMA Refinement 
        # 我们只优化前 3 个最有希望的候选，避免超时
        candidates_to_optimize = top_candidates
        
        self.logger.info(f"[CMA] 开始优化 Top {len(candidates_to_optimize)} 候选...")

        # [修改] 存储所有优化后的动作和分数
        refined_candidates = []

        # L1->CMA 过滤阈值：
        # L1 单次模拟里：犯规约 -500，交换球权约 -100，胜利 +2000，失败 -8000。
        # 因此阈值不宜太“苛刻”，否则 TopK 可能全部被跳过导致 best_score 维持 -inf。
        # 经验上用 -400 左右更贴近“多数回合都是犯规/极不稳定”的水平。
        l1_opt_threshold = -400.0

        # 并行执行所有 CMA 优化
        refined_candidates = []
        
        # 准备并行任务
        cma_tasks = []
        
        # 至少保留 1 个：即使全都低于阈值，也保留 L1 最好的那个进 CMA
        best_l1_idx = None
        if (not is_shooting_8) and candidates_to_optimize:
            best_l1_idx = max(
                range(len(candidates_to_optimize)),
                key=lambda i: candidates_to_optimize[i].get('l1_score', -float('inf')),
            )

        for idx, item in enumerate(candidates_to_optimize):
            raw_action = item['action']
            l1_score = item['l1_score']
            
            # 只有 L1 分数不至于太差的才值得优化；但至少保留 L1 最好的 1 个
            if (not is_shooting_8) and (l1_score < l1_opt_threshold) and (idx != best_l1_idx):
                continue

            cma_tasks.append({
                'initial_action': raw_action,
                'balls': balls,
                'table': table,
                'targets': actual_targets,
                'noise_std': self.noise_std,
                'ball_radius': self.ball_radius,
                'cma_maxiter': self.cma_maxiter,
                'cma_popsize': self.cma_popsize,
                'cma_eval_sims': self.cma_eval_sims,
                'idx': idx,
                'l1_score': l1_score
            })
        
        # 并行提交
        if cma_tasks:
            executor = self._executor
            if executor:
                 cma_results = list(executor.map(worker_cma_optimize_full, cma_tasks))
            else:
                 cma_results = [worker_cma_optimize_full(t) for t in cma_tasks]
            
            # 处理结果并输出日志
            for res in cma_results:
                idx = res['idx']
                refined_action = res['action']
                refined_score = res['score']
                refined_stats = res['stats']
                l1_score = res['l1_score']
                
                if refined_stats:
                    if is_shooting_8:
                        self.logger.info(
                            f"  > 候选 {idx} ({refined_action.get('type', 'unknown')}): "
                            f"L1={l1_score:.1f} -> CMA={refined_score:.1f} "
                            f"(K={refined_stats.get('k')}, win={refined_stats.get('win_rate', 0.0):.2f}, "
                            f"lose={refined_stats.get('lose_rate', 0.0):.2f}, avg_state={refined_stats.get('avg_state', 0.0):.1f})"
                        )
                    else:
                        self.logger.info(
                            f"  > 候选 {idx} ({refined_action.get('type', 'unknown')}): "
                            f"L1={l1_score:.1f} -> CMA={refined_score:.1f} "
                            f"(K={refined_stats.get('k')}, keep={refined_stats.get('keep_rate', 0.0):.2f}, "
                            f"foul={refined_stats.get('foul_rate', 0.0):.2f}, avg_state={refined_stats.get('avg_state', 0.0):.1f})"
                        )
                else:
                    self.logger.info(
                        f"  > 候选 {idx} ({refined_action.get('type', 'unknown')}): L1={l1_score:.1f} -> CMA={refined_score:.1f}"
                    )
                
                refined_candidates.append({
                    'action': refined_action,
                    'score': refined_score,
                    'stats': refined_stats,
                })
        
        # 按分数降序排序
        refined_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. 最终选择
        best_action = None
        best_score = -float('inf')
        
        if is_shooting_8:
            # 黑八阶段：用 CMA 统计到的 win_rate 来选
            if not refined_candidates:
                self.logger.warning("[CueCard] 黑八阶段没有可用候选，切换到安全模式。")
                return self._play_safety(balls, actual_targets, table)

            best_win_rate = -1.0
            best_cma_score = -float('inf')

            for idx, candidate in enumerate(refined_candidates):
                action = candidate['action']
                cma_score = float(candidate.get('score', -float('inf')))
                stats = candidate.get('stats') or {}
                win_rate = stats.get('win_rate', -1.0)
                lose_rate = stats.get('lose_rate', 0.0)
                foul_rate = stats.get('foul_rate', 0.0)
                k = stats.get('k', '?')

                self.logger.info(
                    f"[CueCard] 黑八候选{idx+1}: K={k}, win={win_rate:.2f}, lose={lose_rate:.2f}, foul={foul_rate:.2f} | CMA={cma_score:.1f} | {action.get('type','unknown')}"
                )

                if win_rate > best_win_rate or (win_rate == best_win_rate and cma_score > best_cma_score):
                    best_win_rate = win_rate
                    best_cma_score = cma_score
                    best_action = action
                    best_score = cma_score

            # 硬阈值：黑八进球率必须 >= 0.90，否则宁可防守
            th_rate = 0.90
            if best_win_rate < th_rate or best_action is None:
                self.logger.info(
                    f"[CueCard] 黑八胜率阈值未达标: win_rate={best_win_rate:.2f} < {th_rate}，转为防守。"
                )
                return self._play_safety(balls, actual_targets, table)

            self.logger.info(
                f"[CueCard] 黑八选择: win_rate={best_win_rate:.2f} | CMA={best_cma_score:.1f} | 动作={best_action.get('type','未知')}"
            )
            return best_action

        if refined_candidates:
            # 非黑8阶段，直接选择最佳动作
            best_action = refined_candidates[0]['action']
            best_score = refined_candidates[0]['score']

        # 动态进攻阈值 
        # 依据对手剩余球数动态调整风险偏好
        opp_targets = self._get_opponent_targets(my_targets)
        opp_remaining = len([b for b in opp_targets if b in balls])
        
        # 默认阈值 TODO 阈值合理性检测
        confidence_threshold = 525.0  
        
        # 如果对手快赢了(只剩1-2颗)，我们必须降低门槛拼命
        if opp_remaining <= 2:
            confidence_threshold = 500.0 
        # 如果我们快赢了，稍微稳一点
        elif len(actual_targets) <= 2:
             confidence_threshold = 530.0

        # 如果最佳分数低于阈值，且当前不是必须解球的状态（即没有犯规风险），则防守
        # 注意：best_action 可能是 None
        if best_action is None or best_score < confidence_threshold:
            self.logger.info(f"[Decision] 最佳分数 {best_score:.1f} < 阈值 {confidence_threshold}，转为防守。")
            return self._play_safety(balls, actual_targets, table)

        self.logger.info(f"[CueCard] 执行进攻: {best_action.get('type','未知')} | 预测得分: {best_score:.1f}")
        return best_action


    def _distance_to_nearest_pocket(self, ball_pos, table):
        """计算球到最近袋口的距离（借鉴niceAgent）"""
        return distance_to_nearest_pocket_standalone(ball_pos, table)
    
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
                a = a + np.random.normal(0, self.noise_std['a'])
                b = b + np.random.normal(0, self.noise_std['b'])
            
            # [修复] 确保 a² + b² <= 1.0
            if a**2 + b**2 > 1.0:
                norm = np.sqrt(a**2 + b**2)
                a = a / norm * 0.99
                b = b / norm * 0.99

            # 与 poolenv.py 的动作裁剪保持一致
            V0 = float(np.clip(V0, 0.5, 8.0))
            theta = float(np.clip(theta, 0.0, 90.0))
            a = float(np.clip(a, -0.5, 0.5))
            b = float(np.clip(b, -0.5, 0.5))
            
            cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception as e:
            # 物理模拟失败通常来自 pooltool 内部数值/几何异常或参数/状态不合法。
            # 这里仅记录前几次失败，避免刷屏。
            try:
                cnt = getattr(self, '_simulate_shot_fail_cnt', 0) + 1
                setattr(self, '_simulate_shot_fail_cnt', cnt)
                if cnt <= 3:
                    self.logger.warning(f"[CueCard] simulate失败({cnt}) {type(e).__name__}: {e}")
            except Exception:
                pass
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
        cue_hit_cushion = False
        target_hit_cushion = False
        
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
                ids = getattr(e, 'ids', ())
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_id is not None and first_contact_id in ids:
                    target_hit_cushion = True

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
                else: return True, False, None # 没碰到8直接进了，或者先碰别的球（极少见）
            else:
                return True, False, 'lose' # 还没清台就打进8 = 输

        # 判定 C: 首球犯规
        if first_contact_id is None:
            return True, False, None  # 没有碰到任何球，犯规
        else:
            # 必须先碰到自己的球 (如果是8号球阶段，必须先碰8)
            legal_contacts = my_targets
            if first_contact_id not in legal_contacts:
                return True, False, None  # 犯规
        
        # 判定 D: 吃库 (简化版：无进球且母球与目标球都未吃库 -> 犯规)
        # 说明：这里对齐 "无进球 + (cue未吃库) + (首碰目标球未吃库)" 的判定口径。
        if (not pocketed_ids) and (not cue_hit_cushion) and (not target_hit_cushion):
            return True, False, None

        # 3. 球权判定
        # 合法进球 -> 保留球权
        if own_pocketed:
            return False, True, None
        
        # 没进球 -> 失去球权
        return False, False, None

    def _get_opponent_targets(self, my_targets):
        """推断对手目标球"""
        all_solids = [str(i) for i in range(1, 8)]
        all_stripes = [str(i) for i in range(9, 16)]
        if set(my_targets).intersection(all_solids):
            return all_stripes
        return all_solids

    def _get_break_shot_closest_is_target(self):
        """开球参数（最近球是自己目标球时）。"""
        return {
            "V0": 5.985968056114618,
            "phi": 95.41873839827912,
            "theta": 0.00039084045578199444,
            "a": -0.006044626650933807,
            "b": -0.14417649847771563,
            'type': 'break_target',
        }

    def _get_break_shot_closest_is_nontarget(self):
        """开球参数（最近球不是自己目标球时）。

        说明：这里给一个更偏切/反向加塞的备选，用于在“目标球型已固定”的设置下
        尝试降低首次接触违规球的概率。具体参数可按你的实验再调。
        """
        return {
            "V0": 2.110778676464473,
            "phi": 128.99349641495118,
            "theta": 0.008570750725023295,
            "a": -0.1304418931515131,
            "b": 0.035565863953657434,
            'type': 'break_nontarget',
        }

    def _random_action(self):
        return {
            'V0': random.uniform(1.0, 5.0),
            'phi': random.uniform(0, 360),
            'theta': 0, 'a': 0, 'b': 0 ,'h_prob':0.0,'type':'random'
        }