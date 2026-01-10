
from utils import set_random_seed
from poolenv import PoolEnv
import pooltool as pt
from agents.break_agent import BreakAgent  # Modified import
from agents import BasicAgent
import time
import numpy as np

def evaluate_break(n_episodes=100):
    set_random_seed(42)
    env = PoolEnv()
    agent = BreakAgent.from_json("v3.1.json")
    # agent = BasicAgent()
    
    print(f"开始测试 BreakAgent 开球性能 (共 {n_episodes} 次)...")
    
    # 初始化统计字典
    stats_template = {
        'count': 0,
        'advantage_count': 0,
        'disadvantage_count': 0,
        'even_count': 0,
        'foul_count': 0,
        'gain_control_count': 0,
        'lose_control_count': 0,
        'lose_match_count': 0,
        # Top 1/2/3 Separate Stats
        'my_d1_sum': 0.0, 'my_d1_cnt': 0,
        'my_d2_sum': 0.0, 'my_d2_cnt': 0,
        'my_d3_sum': 0.0, 'my_d3_cnt': 0,
        
        'opp_d1_sum': 0.0, 'opp_d1_cnt': 0,
        'opp_d2_sum': 0.0, 'opp_d2_cnt': 0,
        'opp_d3_sum': 0.0, 'opp_d3_cnt': 0
    }

    import copy
    stats = {
        'total': copy.deepcopy(stats_template),
        'protocol_A': copy.deepcopy(stats_template),
        'protocol_B': copy.deepcopy(stats_template)
    }
    
    # 我们轮换目标球型，模拟不同情况
    target_types = ['solid', 'stripe']
    
    for i in range(n_episodes):
        # 1. 重置环境
        my_suit = target_types[i % 2]
        opp_suit = target_types[(i + 1) % 2]
        env.reset(target_ball=my_suit)
        
        player_id = env.get_curr_player() # 'A' or 'B'
        
        # 2. 获取观察
        obs = env.get_observation(player_id)
        balls_dict = obs[0]
        my_target_ids = obs[1]
        
        # 3. 判定协议类型 (复制 BreakAgent 的逻辑)
        object_balls = [b for bid, b in balls_dict.items() if bid != 'cue']
        is_protocol_A = False
        if object_balls:
            apex_ball = min(object_balls, key=lambda b: b.state.rvw[0][1])
            if apex_ball.id in my_target_ids:
                is_protocol_A = True
        
        current_proto = 'protocol_A' if is_protocol_A else 'protocol_B'
        
        # 3. 决策
        t0 = time.time()
        # Ensure table is passed
        action = agent.decision(obs[0], obs[1], obs[2])
        t1 = time.time()
        
        # 4. 执行
        step_info = env.take_shot(action)
        
        # 5. 分析结果 (Step Info)
        # 检查是否犯规
        is_foul = False
        foul_reason = []
        if step_info.get('WHITE_BALL_INTO_POCKET'):
            is_foul = True
            foul_reason.append("母球洗袋")
        if step_info.get('BLACK_BALL_INTO_POCKET'):
            is_foul = True
            foul_reason.append("黑8进袋")
        if step_info.get('NO_HIT'):
            is_foul = True
            foul_reason.append("未触球")
        if step_info.get('FOUL_FIRST_HIT'):
            is_foul = True
            foul_reason.append("首触犯规")
        if step_info.get('NO_POCKET_NO_RAIL') and not (step_info.get('ME_INTO_POCKET') or step_info.get('ENEMY_INTO_POCKET')):
            is_foul = True
            foul_reason.append("无进球且未吃库")
            
        # 统计
        me_potted = step_info.get('ME_INTO_POCKET', [])
        enemy_potted = step_info.get('ENEMY_INTO_POCKET', [])
        
        me_cnt = len(me_potted)
        opp_cnt = len(enemy_potted)

        # 更新统计
        for key in ['total', current_proto]:
            stats[key]['count'] += 1
            
            

            # --- 新增统计逻辑: 优/劣/平球/犯规 (互斥) ---
            if is_foul:
                stats[key]['foul_count'] += 1
                if key == 'total': break_grade = "犯规"
            else:
                if me_cnt > opp_cnt:
                    stats[key]['advantage_count'] += 1
                    if key == 'total': break_grade = "优球"
                elif me_cnt < opp_cnt:
                    stats[key]['disadvantage_count'] += 1
                    if key == 'total': break_grade = "劣球"
                else:
                    stats[key]['even_count'] += 1
                    if key == 'total': break_grade = "平球"
            # ---------------------------

            # 独立判定各个计数 (允许重叠)
            is_legal_pot = (not is_foul) and (len(me_potted) > 0)
                
            if is_legal_pot:
                stats[key]['gain_control_count'] += 1
                
            if not is_legal_pot:
                stats[key]['lose_control_count'] += 1
                
            is_pot_8 = step_info.get('BLACK_BALL_INTO_POCKET', False)
            if is_pot_8:
                stats[key]['lose_match_count'] += 1
                
            # --- 新增统计逻辑: 距离 (Top 3) ---
            final_balls = step_info.get('BALLS', None)
            if final_balls:
                pocket_centers = [p.center for p in obs[2].pockets.values()]
                
                my_dists = []
                opp_dists = []
                
                for bid, ball in final_balls.items():
                    if bid == 'cue' or bid == '8': continue
                    if ball.state.s == 4: continue # Potted
                    
                    pos = ball.state.rvw[0]
                    min_dist = min([np.linalg.norm(pos - p) for p in pocket_centers])
                    
                    if bid in my_target_ids:
                        my_dists.append(min_dist)
                    else:
                        opp_dists.append(min_dist)
                
                # Sort and store separate Top 1/2/3
                my_dists.sort()
                opp_dists.sort()
                
                # My Balls
                if len(my_dists) > 0:
                    stats[key]['my_d1_sum'] += my_dists[0]
                    stats[key]['my_d1_cnt'] += 1
                if len(my_dists) > 1:
                    stats[key]['my_d2_sum'] += my_dists[1]
                    stats[key]['my_d2_cnt'] += 1
                if len(my_dists) > 2:
                    stats[key]['my_d3_sum'] += my_dists[2]
                    stats[key]['my_d3_cnt'] += 1
                    
                # Opp Balls
                if len(opp_dists) > 0:
                    stats[key]['opp_d1_sum'] += opp_dists[0]
                    stats[key]['opp_d1_cnt'] += 1
                if len(opp_dists) > 1:
                    stats[key]['opp_d2_sum'] += opp_dists[1]
                    stats[key]['opp_d2_cnt'] += 1
                if len(opp_dists) > 2:
                    stats[key]['opp_d3_sum'] += opp_dists[2]
                    stats[key]['opp_d3_cnt'] += 1
            # ---------------------------
            
        # 生成简报字符串
        res_strs = []
        res_strs.append(f"[{'A' if is_protocol_A else 'B'}]")
        res_strs.append(break_grade)
        if is_foul: res_strs.append("犯规")
        
        result_type = " | ".join(res_strs)
                
        # 打印单局简报
        print(f"[{i+1}/{n_episodes}] {result_type} | 决策: {t1-t0:.2f}s | "
              f"Action: V0={action['V0']:.1f}, phi={action['phi']:.1f} | "
              f"My: {len(me_potted)}, Opp: {len(enemy_potted)} | Foul: {','.join(foul_reason)}")

    # 汇总
    print("\n" + "="*60)
    print("开球专项测试结果报告 (分协议统计)")
    print("="*60)
    
    # 定义打印辅助函数
    def print_section(title, key):
        s = stats[key]
        total = s['count']
        if total == 0:
            print(f"--- {title} (无数据) ---")
            return
            
        print(f"--- {title} (共 {total} 次) ---")
        print(f"1. 优球率 (My > Opp):    {s['advantage_count']} ({s['advantage_count']/total*100:.1f}%)")
        print(f"2. 劣球率 (My < Opp):    {s['disadvantage_count']} ({s['disadvantage_count']/total*100:.1f}%)")
        print(f"3. 平球率 (My == Opp):   {s['even_count']} ({s['even_count']/total*100:.1f}%)")
        print(f"4. 犯规率 (Foul):        {s['foul_count']} ({s['foul_count']/total*100:.1f}%)")
        print("-" * 20)
        print(f"5. 获得球权 (Control):   {s['gain_control_count']} ({s['gain_control_count']/total*100:.1f}%)")
        print(f"6. 输掉比赛 (Lose Match): {s['lose_match_count']} ({s['lose_match_count']/total*100:.1f}%)")
        
        def calc_avg(d_sum, cnt):
            return d_sum / cnt if cnt > 0 else 0.0

        my_d1 = calc_avg(s['my_d1_sum'], s['my_d1_cnt'])
        my_d2 = calc_avg(s['my_d2_sum'], s['my_d2_cnt'])
        my_d3 = calc_avg(s['my_d3_sum'], s['my_d3_cnt'])
        
        opp_d1 = calc_avg(s['opp_d1_sum'], s['opp_d1_cnt'])
        opp_d2 = calc_avg(s['opp_d2_sum'], s['opp_d2_cnt'])
        opp_d3 = calc_avg(s['opp_d3_sum'], s['opp_d3_cnt'])
        
        print("-" * 20)
        print(f"7. 己方Top1/2/3距袋: {my_d1:.4f} / {my_d2:.4f} / {my_d3:.4f}")
        print(f"8. 对方Top1/2/3距袋: {opp_d1:.4f} / {opp_d2:.4f} / {opp_d3:.4f}")
        print("")

    print_section("总计 (Total)", 'total')
    print_section("协议 A (己方顶点/顺境)", 'protocol_A')
    print_section("协议 B (对方顶点/逆境)", 'protocol_B')
    
    print("="*60)

if __name__ == "__main__":
    evaluate_break(300)