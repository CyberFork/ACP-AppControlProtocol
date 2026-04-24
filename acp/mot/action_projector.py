"""L→A 空间映射（QLoRA 风格低秩投影）。

将 L 模型最后 token 的 hidden state 映射到 Action Head 的输入空间。

低秩分解：
  proj_A: d_lang → rank  (随机初始化)
  proj_B: rank → d_action (零初始化，训练初期输出为 0)

这样可以在 Stage 1（Perceiver 对齐阶段）冻结 ActionProjector，
Stage 2 再解冻训练，保证训练稳定性。

Args:
    d_lang:   L 模型 hidden size，默认 2048
    d_action: Action Head 输入维度，默认 512
    rank:     低秩秩数，默认 8
"""

import torch
import torch.nn as nn


class ActionProjector(nn.Module):
    """QLoRA 风格低秩投影：语言空间 → 动作空间。

    参数量：d_lang*rank + rank*d_action = 2048*8 + 8*512 = 20,480

    Args:
        d_lang:   语言模型 hidden size，默认 2048
        d_action: Action Head 输入维度，默认 512
        rank:     低秩秩数，默认 8
    """

    def __init__(
        self,
        d_lang: int = 2048,
        d_action: int = 512,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.proj_A = nn.Linear(d_lang, rank, bias=False)
        self.proj_B = nn.Linear(rank, d_action, bias=False)
        nn.init.zeros_(self.proj_B.weight)  # 零初始化，训练初期无动作信号

    def forward(self, last_token_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_token_hidden: L 模型最后 token 的 hidden state [B, d_lang]
        Returns:
            action_embedding: [B, d_action]
        """
        return self.proj_B(self.proj_A(last_token_hidden))
