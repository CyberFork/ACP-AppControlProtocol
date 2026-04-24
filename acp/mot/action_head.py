"""轻量动作预测头（~30M 参数）。

输入：
  instruction_embedding [B, D_l]  — L 模型最后一层最后 token 经 ActionProjector 映射后的向量
  elements: dict with keys:
    'bbox'      [B, N, 4]   — 归一化边界框 (x1,y1,x2,y2)
    'type'      [B, N, 16]  — 元素类型 one-hot（16 种 UI 类型）
    'label_ids' [B, N]      — 元素标签文本 token ids（整数）

输出：
  action_type:    [B, num_action_types]  — [click, type, scroll, press_key, wait] 分类概率
  element_scores: [B, N]                — 每个元素的选择概率
  coord_offset:   [B, 2]                — 相对 bbox 中心的偏移 (dx, dy)，sigmoid 归一化

架构：
  elements → ElementEncoder → [B, N, d_action]
  instruction → Linear(d_action) → query [B, d_action]
  CrossAttention(query, elements) → element_scores [B, N]
  加权求和 → context [B, d_action]
  ActionTypeHead → softmax → [B, 5]
  CoordHead → sigmoid → [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 动作类型枚举（与输出维度对齐）
ACTION_TYPES = ["click", "type", "scroll", "press_key", "wait"]

# ElementEncoder 输入维度：bbox(4) + type_onehot(16) + label_emb(384)
_ELEM_IN_DIM = 4 + 16 + 384


class ActionHead(nn.Module):
    """动作预测头，将指令嵌入和 UI 元素特征映射为可执行动作。

    Args:
        d_lang:          指令嵌入维度（ActionProjector 输出），默认 512
        d_action:        内部动作特征维度，默认 256
        num_action_types: 动作类型数，默认 5
        max_elements:    最大元素数（填充用），默认 64
        label_vocab_size: 元素标签词表大小，默认 8192
        label_emb_dim:   标签嵌入维度，默认 384
    """

    def __init__(
        self,
        d_lang: int = 512,
        d_action: int = 256,
        num_action_types: int = 5,
        max_elements: int = 64,
        label_vocab_size: int = 8192,
        label_emb_dim: int = 384,
    ) -> None:
        super().__init__()
        self.d_action = d_action
        self.max_elements = max_elements

        # 元素标签嵌入
        self.label_embedding = nn.Embedding(label_vocab_size, label_emb_dim)

        # ElementEncoder：拼接特征 → 256d
        self.element_encoder = nn.Sequential(
            nn.Linear(_ELEM_IN_DIM, d_action),
            nn.GELU(),
        )

        # 指令投影：d_lang → d_action
        self.instruction_proj = nn.Linear(d_lang, d_action)

        # 动作类型预测头
        self.action_type_head = nn.Linear(d_action, num_action_types)

        # 坐标偏移预测头
        self.coord_head = nn.Linear(d_action, 2)

    def forward(
        self,
        instruction_embedding: torch.Tensor,
        elements: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            instruction_embedding: [B, d_lang]
            elements: {
                'bbox':      [B, N, 4],
                'type':      [B, N, 16],
                'label_ids': [B, N]  (LongTensor)
            }
        Returns:
            action_type:    [B, num_action_types]  — softmax 概率
            element_scores: [B, N]                 — softmax 概率
            coord_offset:   [B, 2]                 — sigmoid 归一化偏移
        """
        bbox = elements["bbox"]          # [B, N, 4]
        elem_type = elements["type"]     # [B, N, 16]
        label_ids = elements["label_ids"]  # [B, N]

        # 标签嵌入
        label_emb = self.label_embedding(label_ids)  # [B, N, 384]

        # 拼接元素特征
        elem_feats = torch.cat([bbox, elem_type, label_emb], dim=-1)  # [B, N, 404]

        # 编码元素
        elem_enc = self.element_encoder(elem_feats)  # [B, N, d_action]

        # 投影指令为 query
        query = self.instruction_proj(instruction_embedding)  # [B, d_action]

        # Cross-attention：query 关注每个元素
        # scores = Q · K^T / sqrt(d)，Q=[B,1,d], K=[B,N,d]
        scale = self.d_action ** 0.5
        scores = torch.bmm(
            query.unsqueeze(1), elem_enc.transpose(1, 2)
        ).squeeze(1) / scale  # [B, N]
        element_scores = F.softmax(scores, dim=-1)  # [B, N]

        # 加权聚合上下文
        context = torch.bmm(
            element_scores.unsqueeze(1), elem_enc
        ).squeeze(1)  # [B, d_action]

        # 预测动作类型
        action_type = F.softmax(self.action_type_head(context), dim=-1)  # [B, 5]

        # 预测坐标偏移（sigmoid 归一化到 [0,1]）
        coord_offset = torch.sigmoid(self.coord_head(context))  # [B, 2]

        return action_type, element_scores, coord_offset
