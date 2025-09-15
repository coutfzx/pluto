import torch
import torch.nn as nn

from ..layers.mlp_layer import MLPLayer

# 目的: AgentPredictor 的核心任务是为场景中的其他车辆（Agent）预测它们未来的运动轨迹。
# 输入: 它接收的是已经由编码器（如 AgentEncoder 和 Transformer Encoder）处理过的、包含了 Agent 当前状态及其环境上下文信息的高维特征向量。
# 结构: 采用了多任务学习的思想，使用三个独立的 MLP（MLPLayer） 作为“预测头”，分别预测位置、朝向和速度。这种设计简化了模型，允许每个属性独立学习其映射关系。
# 输出: 最终输出一个结构化的张量，包含了每个被预测 Agent 在未来所有时间步的位置、朝向（以 sin/cos 表示）、速度信息。

class AgentPredictor(nn.Module):
    def __init__(self, dim, future_steps) -> None:
        super().__init__()

        self.future_steps = future_steps

        self.loc_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)
        self.yaw_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)
        self.vel_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)

    def forward(self, x):
        """
        x: (bs, N, dim)
        """

        bs, N, _ = x.shape

        loc = self.loc_predictor(x).view(bs, N, self.future_steps, 2)
        yaw = self.yaw_predictor(x).view(bs, N, self.future_steps, 2)
        vel = self.vel_predictor(x).view(bs, N, self.future_steps, 2)

        prediction = torch.cat([loc, yaw, vel], dim=-1)
        return prediction
