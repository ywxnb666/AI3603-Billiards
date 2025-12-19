"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

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
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """几何瞄准与增强优化策略 Agent (Plan B)
    
    核心思路：
    1. 使用几何计算确定瞄准点和击球角度
    2. 评估每个目标球的可行性（直线入袋）
    3. 结合贝叶斯优化微调击球参数
    4. 考虑白球走位，为下一杆做准备
    """
    
    def __init__(self):
        super().__init__()
        
        # 球桌参数（标准8球台球桌）
        self.table_length = 2.54  # 桌面长度 (m)
        self.table_width = 1.27   # 桌面宽度 (m)
        self.ball_radius = 0.028575  # 球半径 (m)
        self.pocket_radius = 0.06  # 球袋半径 (m)
        
        # 球袋位置（6个袋口）
        self.pockets = {
            'top_left': np.array([0.0, self.table_width]),
            'top_middle': np.array([self.table_length / 2, self.table_width]),
            'top_right': np.array([self.table_length, self.table_width]),
            'bottom_left': np.array([0.0, 0.0]),
            'bottom_middle': np.array([self.table_length / 2, 0.0]),
            'bottom_right': np.array([self.table_length, 0.0])
        }
        
        # 优化参数
        self.GEOMETRIC_SAMPLES = 15  # 几何采样数
        self.FINE_TUNE_SAMPLES = 8   # 精细调优采样数
        
        print("NewAgent (几何瞄准与增强优化) 已初始化。")
    
    def _get_ball_position(self, ball):
        """获取球的2D位置坐标"""
        return np.array([ball.state.rvw[0][0], ball.state.rvw[0][1]])
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的距离"""
        return np.linalg.norm(pos2 - pos1)
    
    def _calculate_angle(self, from_pos, to_pos):
        """计算从from_pos指向to_pos的角度（度数，0-360）"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg % 360
    
    def _is_path_clear(self, start_pos, end_pos, balls, ignore_ids):
        """检查从start_pos到end_pos的路径是否畅通（无其他球阻挡）
        
        参数：
            start_pos: 起始位置
            end_pos: 目标位置
            balls: 所有球的字典
            ignore_ids: 忽略的球ID列表
        
        返回：
            bool: True表示路径畅通
        """
        path_vec = end_pos - start_pos
        path_length = np.linalg.norm(path_vec)
        
        if path_length < 1e-6:
            return True
        
        path_dir = path_vec / path_length
        
        # 检查其他球是否阻挡路径
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4:  # 跳过指定球和已入袋的球
                continue
            
            ball_pos = self._get_ball_position(ball)
            
            # 计算球心到路径的垂直距离
            start_to_ball = ball_pos - start_pos
            projection = np.dot(start_to_ball, path_dir)
            
            # 球是否在路径的延伸范围内
            if projection < -self.ball_radius or projection > path_length + self.ball_radius:
                continue
            
            # 计算垂直距离
            perp_dist = np.linalg.norm(start_to_ball - projection * path_dir)
            
            # 如果距离小于两个球半径，则路径被阻挡
            if perp_dist < 2 * self.ball_radius * 1.1:  # 1.1为安全系数
                return False
        
        return True
    
    def _calculate_aim_point(self, cue_pos, target_pos, pocket_pos):
        """计算瞄准点（目标球背后的点，使目标球沿着朝向袋口的方向被击打）
        
        原理：要让目标球进入袋口，白球应该击打目标球与袋口连线的反方向点
        """
        # 从袋口指向目标球的方向向量
        pocket_to_target = target_pos - pocket_pos
        distance = np.linalg.norm(pocket_to_target)
        
        if distance < 1e-6:
            return None
        
        # 归一化
        direction = pocket_to_target / distance
        
        # 瞄准点：目标球背后一个球半径的位置
        aim_point = target_pos + direction * (2 * self.ball_radius)
        
        return aim_point
    
    def _evaluate_shot_geometry(self, cue_pos, target_ball_id, target_pos, pocket_pos, balls):
        """几何评估一次击球的质量
        
        返回：
            dict: {
                'feasible': bool,  # 是否可行
                'aim_point': np.array,  # 瞄准点
                'angle': float,  # 击球角度
                'distance': float,  # 距离
                'cut_angle': float,  # 切球角度
                'score': float  # 综合得分
            }
        """
        result = {
            'feasible': False,
            'aim_point': None,
            'angle': 0,
            'distance': 0,
            'cut_angle': 0,
            'score': -1000
        }
        
        # 计算瞄准点
        aim_point = self._calculate_aim_point(cue_pos, target_pos, pocket_pos)
        if aim_point is None:
            return result
        
        result['aim_point'] = aim_point
        
        # 检查白球到瞄准点的路径是否畅通
        if not self._is_path_clear(cue_pos, aim_point, balls, ignore_ids=['cue', target_ball_id]):
            return result
        
        # 检查目标球到袋口的路径是否畅通
        if not self._is_path_clear(target_pos, pocket_pos, balls, ignore_ids=['cue', target_ball_id]):
            return result
        
        # 计算击球角度和距离
        result['angle'] = self._calculate_angle(cue_pos, aim_point)
        result['distance'] = self._calculate_distance(cue_pos, aim_point)
        
        # 计算切球角度（白球-瞄准点方向 与 目标球-袋口方向 的夹角）
        cue_to_aim = aim_point - cue_pos
        target_to_pocket = pocket_pos - target_pos
        
        if np.linalg.norm(cue_to_aim) > 1e-6 and np.linalg.norm(target_to_pocket) > 1e-6:
            cue_to_aim_norm = cue_to_aim / np.linalg.norm(cue_to_aim)
            target_to_pocket_norm = target_to_pocket / np.linalg.norm(target_to_pocket)
            
            cos_angle = np.dot(cue_to_aim_norm, target_to_pocket_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            cut_angle = math.degrees(math.acos(cos_angle))
            result['cut_angle'] = cut_angle
        else:
            return result
        
        # 标记为可行
        result['feasible'] = True
        
        # 综合评分（距离越近越好，切球角度越小越好，袋口距离越近越好）
        distance_to_pocket = self._calculate_distance(target_pos, pocket_pos)
        
        score = 100
        score -= result['distance'] * 5  # 距离惩罚
        score -= result['cut_angle'] * 0.5  # 切球角度惩罚（角度越大越难）
        score -= distance_to_pocket * 10  # 目标球到袋口的距离惩罚
        
        # 直球加分（切球角度接近0度）
        if result['cut_angle'] < 15:
            score += 20
        
        result['score'] = score
        
        return result
    
    def _find_best_shot(self, balls, my_targets, table):
        """寻找最佳击球方案
        
        返回：
            dict: 最佳击球的几何信息，如果没有可行方案则返回None
        """
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return None
        
        cue_pos = self._get_ball_position(cue_ball)
        
        # 遍历所有目标球和袋口组合，找到最佳方案
        best_shot = None
        best_score = -float('inf')
        
        for target_id in my_targets:
            target_ball = balls.get(target_id)
            if target_ball is None or target_ball.state.s == 4:
                continue
            
            target_pos = self._get_ball_position(target_ball)
            
            # 尝试所有袋口
            for pocket_name, pocket_pos in self.pockets.items():
                shot_eval = self._evaluate_shot_geometry(
                    cue_pos, target_id, target_pos, pocket_pos, balls
                )
                
                if shot_eval['feasible'] and shot_eval['score'] > best_score:
                    best_score = shot_eval['score']
                    best_shot = {
                        'target_id': target_id,
                        'pocket_name': pocket_name,
                        'pocket_pos': pocket_pos,
                        **shot_eval
                    }
        
        return best_shot
    
    def _geometric_to_action(self, best_shot, balls):
        """将几何击球方案转换为动作参数"""
        if best_shot is None:
            return None
        
        # 基础参数
        action = {
            'phi': best_shot['angle'],
            'theta': 0,  # 水平击球
            'a': 0,      # 击打球心
            'b': 0       # 击打球心
        }
        
        # 根据距离调整速度
        distance = best_shot['distance']
        if distance < 0.5:
            action['V0'] = 1.5
        elif distance < 1.0:
            action['V0'] = 2.5
        elif distance < 1.5:
            action['V0'] = 3.5
        else:
            action['V0'] = 4.5
        
        # 根据切球角度微调
        if best_shot['cut_angle'] > 30:
            action['V0'] *= 1.2  # 困难切球需要更大力量
        
        # 限制速度范围
        action['V0'] = np.clip(action['V0'], 0.5, 8.0)
        
        return action
    
    def decision(self, balls=None, my_targets=None, table=None):
        """使用几何瞄准与优化策略进行决策
        
        参数：
            balls: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回：
            dict: 击球动作
        """
        if balls is None:
            print("[NewAgent] 未收到balls信息，使用随机动作。")
            return self._random_action()
        
        try:
            # 检查是否需要打黑8
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[NewAgent] 目标球已清空，切换目标为黑8")
            
            # 1. 使用几何方法寻找最佳击球方案
            best_shot = self._find_best_shot(balls, my_targets, table)
            
            if best_shot is None:
                print("[NewAgent] 未找到可行的几何击球方案，使用随机动作。")
                return self._random_action()
            
            print(f"[NewAgent] 最佳方案: 目标球{best_shot['target_id']} -> "
                  f"{best_shot['pocket_name']}袋, "
                  f"角度{best_shot['angle']:.1f}°, "
                  f"距离{best_shot['distance']:.2f}m, "
                  f"切角{best_shot['cut_angle']:.1f}°, "
                  f"得分{best_shot['score']:.1f}")
            
            # 2. 转换为动作参数
            action = self._geometric_to_action(best_shot, balls)
            
            if action is None:
                print("[NewAgent] 动作转换失败，使用随机动作。")
                return self._random_action()
            
            # 3. 使用物理模拟微调参数
            action = self._fine_tune_action(action, balls, my_targets, table)
            
            print(f"[NewAgent] 最终决策: V0={action['V0']:.2f}, "
                  f"phi={action['phi']:.2f}, theta={action['theta']:.2f}, "
                  f"a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[NewAgent] 决策时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
    
    def _fine_tune_action(self, base_action, balls, my_targets, table):
        """使用物理模拟对动作参数进行微调
        
        参数：
            base_action: 基础动作参数
            balls: 球状态
            my_targets: 目标球列表
            table: 球桌
        
        返回：
            dict: 优化后的动作参数
        """
        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 定义微调范围
            v0_range = [max(0.5, base_action['V0'] - 1.0), min(8.0, base_action['V0'] + 1.0)]
            phi_range = [(base_action['phi'] - 5) % 360, (base_action['phi'] + 5) % 360]
            
            best_action = base_action.copy()
            best_score = -float('inf')
            
            # 采样测试
            for _ in range(self.FINE_TUNE_SAMPLES):
                test_action = {
                    'V0': np.random.uniform(v0_range[0], v0_range[1]),
                    'phi': np.random.uniform(phi_range[0], phi_range[1]) % 360,
                    'theta': base_action['theta'] + np.random.uniform(-2, 2),
                    'a': base_action['a'] + np.random.uniform(-0.05, 0.05),
                    'b': base_action['b'] + np.random.uniform(-0.05, 0.05)
                }
                
                # 限制范围
                test_action['theta'] = np.clip(test_action['theta'], 0, 90)
                test_action['a'] = np.clip(test_action['a'], -0.5, 0.5)
                test_action['b'] = np.clip(test_action['b'], -0.5, 0.5)
                
                # 模拟测试
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    shot.cue.set_state(
                        V0=test_action['V0'],
                        phi=test_action['phi'],
                        theta=test_action['theta'],
                        a=test_action['a'],
                        b=test_action['b']
                    )
                    
                    if not simulate_with_timeout(shot, timeout=2):
                        continue
                    
                    score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                    
                    if score > best_score:
                        best_score = score
                        best_action = test_action.copy()
                        
                except Exception:
                    continue
            
            if best_score > -float('inf'):
                return best_action
            else:
                return base_action
                
        except Exception as e:
            print(f"[NewAgent] 微调失败: {e}")
            return base_action