```python
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, 
                src_pad_idx,
                trg_pad_idx,
                enc_voc_size,
                dec_voc_size,
                d_model,
                n_heads,
                ffn_hidden,
                max_len,
                n_layers,
                drop_prob,
                device):
        super().__init__()
        
        # Encoder：把源序列编码为上下文表示（memory）
        self.encoder = Encoder(
            enc_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device
        )
        # Decoder：自注意力 + 交叉注意力，基于 memory 生成目标序列
        self.decoder = Decoder(
            dec_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        """
        B：batch size，批大小（一次并行处理的样本数）
        H：num_heads，注意力头的数量
        Lq：query 序列长度（查询序列的时间步数/token 数）
        Lk：key 序列长度（键/值序列的时间步数/token 数）
        返回：布尔掩码，形状 (B, 1, Lq, Lk)，True 表示“保留/可见”。
        """
        len_q, len_k = q.size(1), k.size(1)
        # 序列有效位（非 pad）
        q_mask = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)  # (B,1,Lq,1)
        k_mask = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)  # (B,1,1,Lk)
        # 广播得到配对可见性 (B,1,Lq,Lk)
        mask = q_mask & k_mask
        return mask

    def make_causal_mask(self, q, k):
        """
        因果掩码：下三角 True 表示允许可见（仅看见当前及之前位置）。
        返回形状 (1, 1, Lq, Lk)，便于在 (B,H,Lq,Lk) 上广播。
        """
        Lq, Lk = q.size(1), k.size(1)
        mask_2d = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool, device=self.device))  # (Lq,Lk)
        return mask_2d.unsqueeze(0).unsqueeze(1)  # (1,1,Lq,Lk)

    def forward(self, src, trg):
        # 为编码器自注意力生成 padding 掩码。
        # 因为是自注意力，查询序列和键序列都是 src，所以两次都传 src 和相同的 pad 索引。
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)           # (B,1,Ls,Ls)
        # 目标侧自注意力 = padding 掩码 & 因果掩码（都为 True=可见 的布尔掩码）
        trg_mask = (
            self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)                   # (B,1,Lt,Lt)
            & self.make_causal_mask(trg, trg)                                                  # (1,1,Lt,Lt)
        )

        # 假设 Encoder/Decoder 已实现并接受 (x, mask)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, trg_mask, src_mask)
        return out


    
