
import multiprocessing
import cma
import numpy as np
from poolenv import PoolEnv
from agents import BreakAgent
import time
import json
import os
import logging

def _setup_worker():
    """Worker process initialization"""
    import os
    import time
    # 1. Reset RNG seed
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    np.random.seed(seed)
    # 2. Suppress poolenv logs
    logging.getLogger('poolenv').setLevel(logging.WARNING)

def _get_apex_type(env, my_suit):
    """Helper to detect if current state is Friendly (A) or Hostile (B)"""
    player_id = env.get_curr_player()
    obs = env.get_observation(player_id)
    balls = obs[0]
    my_targets = obs[1]
    
    object_balls = [b for bid, b in balls.items() if bid != 'cue']
    if not object_balls: return None
    
    apex_ball = min(object_balls, key=lambda b: b.state.rvw[0][1])
    is_friendly = (apex_ball.id in my_targets)
    return 'A' if is_friendly else 'B'


def calculate_reward_A(step_info, balls_dict, my_targets, table):
    """Reward function for Protocol A (Aggressive)"""
    total_score = 0.0
    
    # 1. 致命错误：黑八进袋
    if step_info.get('BLACK_BALL_INTO_POCKET'):
        return -10000.0 # 提高惩罚权重
        
    # 2. 犯规判定
    # 你的规则：白球进袋 或 首触非己方 -> 恢复状态。这对A情况来说是不可接受的失败。
    vals = ['WHITE_BALL_INTO_POCKET', 'NO_HIT', 'FOUL_FIRST_HIT', 'NO_POCKET_NO_RAIL']
    if any(step_info.get(k) for k in vals):
        return -100.0 # 直接返回，犯规就没有后续的空间奖励了
        
    # 3. 核心目标：连杆奖励 (Rule 3)
    # 只要有己方球进袋，就是巨大的成功，因为获得了下一杆机会
    me_potted_count = len(step_info.get('ME_INTO_POCKET', []))
    opp_potted_count = len(step_info.get('ENEMY_INTO_POCKET', []))
    
    if me_potted_count > 0:
        total_score += 200.0 # [重要] 巨大的阶跃奖励，区分"进球"与"没进球"
        total_score += me_potted_count * 50.0 # 每多进一个额外加分
    
    # 4. 局势判断：净胜球
    total_score += (me_potted_count - opp_potted_count) * 20.0
    
    # 5. 空间奖励：炸散程度 (Spread) 而非单纯的距离袋口
    # 对于开球，球堆炸得越散，后续清台越容易。
    # 计算所有球位置的方差或平均成对距离
    current_balls = step_info.get('BALLS', balls_dict)
    ball_positions = []
    for bid, ball in current_balls.items():
        if bid != 'cue' and ball.state.s != 4: # 不在袋中的球
             ball_positions.append(ball.state.rvw[0])
    
    if ball_positions:
        # 计算所有球相对于球桌中心的扩散程度 (Dispersion)
        # 假设球桌中心是 [table_width/2, table_length/2]
        # 或者简化为：计算所有球坐标的标准差
        positions_np = np.array(ball_positions)
        spread_score = np.sum(np.std(positions_np, axis=0)) 
        # std越大，球越散。通常会在 0.x 到 1.x 之间
        total_score += spread_score * 50.0 

    return total_score

def calculate_reward_B(step_info, balls_dict, opp_targets, table):
    """Reward function for Protocol B (Precision/Safety)"""
    total_score = 0.0
    
    # 1. 致命错误
    if step_info.get('BLACK_BALL_INTO_POCKET'):
        return -10000.0
        
    # 2. 规则4特化惩罚：首触错误
    # 在情况B，这是最容易踩的坑
    if step_info.get('FOUL_FIRST_HIT'):
        return -500.0 # 极大的惩罚，告诉Agent：绝对不要碰顶点球！
        
    if step_info.get('WHITE_BALL_INTO_POCKET') or step_info.get('NO_HIT'):
        return -200.0

    # 3. 合法接触奖励 (Legal Hit Bonus)
    # 如果没有犯规，说明Agent成功避开了顶点球S，击中了己方球F。
    # 这在情况B中本身就是一种由于几何结构带来的巨大成功。
    total_score += 100.0 
    
    # 4. 进球奖励
    me_potted_count = len(step_info.get('ME_INTO_POCKET', []))
    opp_potted_count = len(step_info.get('ENEMY_INTO_POCKET', []))
    
    if me_potted_count > 0:
        total_score += 300.0 # 进球更是难上加难，给予更高奖励
    
    # 5. 防守位奖励 (Safety/Snooker)
    # 如果没进球（交换球权），我们希望给对方留一个烂摊子。
    # 你的原代码逻辑是“对方球离袋口远”，这在B情况是合理的。
    # 也可以加入：白球停留在远离球堆的地方，或者白球贴库。
    
    # 沿用并增强你的"对方球远离袋口"逻辑
    BALL_DIAMETER = 0.05715
    SAFE_DIST = 4 * BALL_DIAMETER
    
    pocket_centers = []
    if table and getattr(table, 'pockets', None):
         pocket_centers = [p.center for p in table.pockets.values()]
         
    current_balls = step_info.get('BALLS', balls_dict)
    
    dist_score = 0
    opp_ball_count = 0
    for ball_id in opp_targets: # 这里的opp_targets实际上是对方的花色
        if ball_id == '8': continue
        ball = current_balls.get(ball_id)
        if not ball or ball.state.s == 4: continue # 已进袋或不存在
        
        opp_ball_count += 1
        pos = ball.state.rvw[0]
        min_dist = float('inf')
        for p_center in pocket_centers:
            # p_center is numpy array
            dist = np.linalg.norm(pos - p_center)
            if dist < min_dist:
                min_dist = dist
        
        # 对方球离袋口越远越好
        if min_dist > SAFE_DIST:
            dist_score += 1.0
        else:
            dist_score += (min_dist / SAFE_DIST)
            
    if opp_ball_count > 0:
        total_score += (dist_score / opp_ball_count) * 30.0 # 归一化后加权

    # 6. 白球贴库奖励 (Cue Ball Cushion Safety)
    cue_ball = current_balls.get('cue')
    if cue_ball and cue_ball.state.s != 4:
        # Assuming table is centered
        w = table.w
        l = table.l
        pos = cue_ball.state.rvw[0] # [x, y, z]
        
        # Dist to x-walls (+/- w/2)
        dist_x = (w / 2.0) - abs(pos[0])
        # Dist to y-walls (+/- l/2)
        dist_y = (l / 2.0) - abs(pos[1])
        
        min_cushion_dist = min(dist_x, dist_y)
        
        # 距离越近越好
        if min_cushion_dist < SAFE_DIST:
            # 0 dist -> 1.0
            # SAFE_DIST -> 0.0
            cushion_score = 1.0 - (min_cushion_dist / SAFE_DIST)
            total_score += cushion_score * 30.0

    return total_score

def evaluate_proto_A(vec_A, n_episodes=10):
    """Evaluate Protocol A parameters. Loops until 'A' scenario is found."""
    _setup_worker()
    
    # Construct partial params for A
    params = {
        'A_V0': float(np.clip(vec_A[0], 5.0, 8.0)),
        'A_phi': float(vec_A[1]), # No clip? Assuming 0-360 handled by trig or loose
        'A_theta': float(np.clip(vec_A[2], 0.0, 80.0)),
        'A_a': float(np.clip(vec_A[3], -0.5, 0.5)),
        'A_b': float(np.clip(vec_A[4], -0.5, 0.5))
    }
    # Note: B params don't matter here as we only run A
    
    env = PoolEnv()
    # Force noise settings
    env.noise_std = {'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005}
    env.enable_noise = True
    
    agent = BreakAgent(params=params) # Load A params
    
    total_score = 0.0
    completed = 0
    target_types = ['solid', 'stripe']
    attempt = 0
    
    while completed < n_episodes:
        attempt += 1
        # Random setup
        my_suit = target_types[attempt % 2]
        env.reset(target_ball=my_suit)
        
        # Check type
        proto_type = _get_apex_type(env, my_suit)
        if proto_type != 'A':
            continue # Skip B scenarios
            
        # Execute A
        player_id = env.get_curr_player()
        obs = env.get_observation(player_id)
        # obs = (balls, my_targets, table)
        balls = obs[0]
        my_targets = obs[1]
        table = obs[2]
        
        try:
            # Agent will see is_friendly=True and use params['A_*']
            action = agent.decision(balls, my_targets, table)
            step_info = env.take_shot(action)
            
            # Pass new args to reward function
            r = calculate_reward_A(step_info, balls, my_targets, table)
            total_score += r
            completed += 1
        except Exception as e:
            print(f"Error in A eval: {e}")
            total_score -= 100
            completed += 1
            
    return -(total_score / n_episodes)

def evaluate_proto_B(vec_B, n_episodes=20):
    """Evaluate Protocol B parameters. Loops until 'B' scenario is found."""
    _setup_worker()
    
    params = {
        'B_V0': float(np.clip(vec_B[0], 1.0, 5.0)),
        'B_phi': float(vec_B[1]),
        'B_theta': float(np.clip(vec_B[2], 0.0, 80.0)),
        'B_a': float(np.clip(vec_B[3], -0.5, 0.5)),
        'B_b': float(np.clip(vec_B[4], -0.5, 0.5))
    }
    
    env = PoolEnv()
    env.noise_std = {'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005}
    env.enable_noise = True
    agent = BreakAgent(params=params)
    
    total_score = 0.0
    completed = 0
    target_types = ['solid', 'stripe']
    attempt = 0
    
    while completed < n_episodes:
        attempt += 1
        my_suit = target_types[attempt % 2]
        env.reset(target_ball=my_suit)
        
        if _get_apex_type(env, my_suit) != 'B':
            continue
            
        player_id = env.get_curr_player()
        obs = env.get_observation(player_id)
        balls = obs[0]
        # For B, my_targets are my balls, but I want to keep OPPONENT balls safe?
        # Protocol B goal: "detect all non-target ... balls"
        # If I am B, my targets are 9-15. Non-targets are 1-7.
        # "my_targets" in obs is MY balls.
        # So "non-target" = set(all_balls) - set(my_targets) - set(cue, 8)
        # Construct opponent targets list
        my_targets = obs[1]
        all_ids = [str(i) for i in range(1, 16)]
        opp_targets = [bid for bid in all_ids if bid not in my_targets and bid != '8']
        
        table = obs[2]
        
        try:
            action = agent.decision(balls, my_targets, table)
            step_info = env.take_shot(action)
            r = calculate_reward_B(step_info, balls, opp_targets, table)
            total_score += r
            completed += 1
        except Exception as e:
            print(f"Error in B eval: {e}")
            total_score -= 100
            completed += 1
            
    return -(total_score / n_episodes)

def train():
    print("开始双协议 CMA-ES 优化 (Protocol A & B)...")
    
    # 1. Setup ES for A
    # A: [V0, phi, theta, a, b]
    x0_A = [
        5.985968056114618, 
        95.41873839827912, 
        0.00039084045578199444,
        -0.006044626650933807, 
        -0.14417649847771563,
    ] # Initial guess from prev best
    sigma_A = 0.01 # Learning rate for A
    opts_A = {
        'maxiter': 50, 'popsize': 8, 'seed': 42,
        'bounds': [[5.0, 80.0, 0.0, -0.5, -0.5], [8.0, 110.0, 30.0, 0.5, 0.5]] # Phi restricted to ~90
    }
    es_A = cma.CMAEvolutionStrategy(x0_A, sigma_A, opts_A)
    
    # 2. Setup ES for B
    # B: [V0, phi, theta, a, b]
    x0_B = [
        2.110778676464473, 
        128.99349641495118, 
        0.008570750725023295, 
        -0.1304418931515131,
        0.035565863953657434,
    ] # Initial guess
    sigma_B = 0.01
    opts_B = {
        'maxiter': 50, 'popsize': 8, 'seed': 43,
        'bounds': [[1.0, 110.0, 0.0, -0.5, -0.5], [5.0, 150.0, 30.0, 0.5, 0.5]] # Phi restricted to ~130
    }
    es_B = cma.CMAEvolutionStrategy(x0_B, sigma_B, opts_B)
    
    best_loss_A = float('inf')
    best_vec_A = x0_A
    best_loss_B = float('inf')
    best_vec_B = x0_B
    
    with multiprocessing.Pool(processes=4) as pool:
        # Loop until both are done (assuming same maxiter)
        while not (es_A.stop() and es_B.stop()):
            
            # --- Optimize A ---
            if not es_A.stop():
                sols_A = es_A.ask()
                # Run parallel eval for A
                fit_A = pool.map(evaluate_proto_A, sols_A)
                es_A.tell(sols_A, fit_A)
                print(f"\n--- Generation A {es_A.countiter} --- Sigma: {es_A.sigma:.10f}")
                es_A.disp()
                
                # Dynamic Learning Rate Decay
                if es_A.countiter > 0 and es_A.countiter % 10 == 0:
                    print(f"Decaying sigma_A: {es_A.sigma:.10f} -> {es_A.sigma * 0.75:.10f}")
                    es_A.sigma *= 0.75
                
                if es_A.result.fbest < best_loss_A:
                    best_loss_A = es_A.result.fbest
                    best_vec_A = es_A.result.xbest
            
            # --- Optimize B ---
            if not es_B.stop():
                sols_B = es_B.ask()
                # Run parallel eval for B
                fit_B = pool.map(evaluate_proto_B, sols_B)
                es_B.tell(sols_B, fit_B)
                print(f"\n--- Generation B {es_B.countiter} --- Sigma: {es_B.sigma:.10f}")
                es_B.disp()
                
                # Dynamic Learning Rate Decay
                if es_B.countiter > 0 and es_B.countiter % 10 == 0:
                    print(f"Decaying sigma_B: {es_B.sigma:.10f} -> {es_B.sigma * 0.75:.10f}")
                    es_B.sigma *= 0.75
                
                if es_B.result.fbest < best_loss_B:
                    best_loss_B = es_B.result.fbest
                    best_vec_B = es_B.result.xbest
            
            # Save intermediate
            save_combined_params(best_vec_A, best_vec_B, "best_params_dual_cma.json")

    print("\n优化完成!")
    save_combined_params(best_vec_A, best_vec_B, "best_params_dual_final.json")

def save_combined_params(vec_A, vec_B, filename):
    data = {
        'A_V0': float(vec_A[0]), 'A_phi': float(vec_A[1]), 
        'A_theta': float(vec_A[2]), 'A_a': float(vec_A[3]), 'A_b': float(vec_A[4]),
        
        'B_V0': float(vec_B[0]), 'B_phi': float(vec_B[1]), 
        'B_theta': float(vec_B[2]), 'B_a': float(vec_B[3]), 'B_b': float(vec_B[4])
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Params saved to {filename}")

if __name__ == "__main__":
    train()
