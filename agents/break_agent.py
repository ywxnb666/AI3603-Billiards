
import math
import numpy as np
import random
import pooltool as pt

class BreakAgent:
    def __init__(self, params=None):
        self.ball_radius = 0.028575
        self.table_width = 0.9906
        self.table_length = 1.9812
        
        # 默认策略参数 (10个参数)
        # 协议A: 己方顶点 (High Velocity, ~90 deg)
        # 协议B: 对方顶点 (Low Velocity, ~130 deg for kick shot)
        default_params = {
            # Protocol A Params
            'A_V0': 7.12, 'A_phi': 95.5, 'A_theta': 0.0, 'A_a': 0.0, 'A_b': -0.26,
            # Protocol B Params
            'B_V0': 2.14, 'B_phi': 129.6, 'B_theta': 0.0, 'B_a': 0.0, 'B_b': -0.19
        }
        
        self.params = default_params
        if params is not None:
            self.params.update(params)

    @classmethod
    def from_json(cls, json_path):
        """从JSON文件加载参数并创建实例"""
        import json
        with open(json_path, 'r') as f:
            params = json.load(f)
        return cls(params=params)

    def decision(self, balls, my_targets, table=None):
        """
        主要决策方法。
        根据顶点球关系选择协议A或B，并执行对应的参数。
        """
        cue_ball = balls.get('cue')
        if not cue_ball:
            return self._random_action()
            
        # 寻找顶点球（所有目标球中 Y 坐标最小的）
        object_balls = [b for bid, b in balls.items() if bid != 'cue']
        if not object_balls:
             return self._random_action()
             
        apex_ball = min(object_balls, key=lambda b: b.state.rvw[0][1])
        apex_id = apex_ball.id
        
        # 确定状态: 顺境(A) 或 逆境(B)
        is_friendly = (apex_id in my_targets)
        
        print(f"[BreakAgent] 顶点球 ID: {apex_id}, 花色匹配: {is_friendly}")
        
        if is_friendly:
            return self._execute_protocol_A()
        else:
            return self._execute_protocol_B()
            
    def _execute_protocol_A(self):
        print("[BreakAgent] 执行协议 A：己方顶点")
        return {
            'V0': self.params['A_V0'],
            'phi': self.params['A_phi'],
            'theta': self.params['A_theta'],
            'a': self.params['A_a'],
            'b': self.params['A_b']
        }
        
    def _execute_protocol_B(self):
        print("[BreakAgent] 执行协议 B：对方顶点")
        return {
            'V0': self.params['B_V0'],
            'phi': self.params['B_phi'],
            'theta': self.params['B_theta'],
            'a': self.params['B_a'],
            'b': self.params['B_b']
        }

    def _random_action(self):
        return {'V0': 1.0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0}
