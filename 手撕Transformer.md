# 完整Transformer
自注意力部分为MHA
```python
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦-余弦位置编码（batch_first 版本）。
    输入/输出: (B, L, D)
    """

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        length = x.size(1)
        x = x + self.pe[:, :length, :]
        return self.dropout(x)


class MHA(nn.Module):
    """
    - __init__(head_num, dimension_k, dimension_v, d_k, d_v, d_o, dropout)
    - forward(q, k, v, mask) → 返回 (attn, out)
    约定：mask 为加性掩码，形状可广播到 (B, 1, Lq, Lk)，屏蔽处为 -inf，允许处为 0。
    """

    def __init__(self, head_num: int, dimension_k: int, dimension_v: int, d_k: int, d_v: int, d_o: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head_num = head_num
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o

        self.fc_q = nn.Linear(dimension_k, head_num * d_k)
        self.fc_k = nn.Linear(dimension_k, head_num * d_k)
        self.fc_v = nn.Linear(dimension_v, head_num * d_v)

        self.dropout = nn.Dropout(dropout)
        self.fc_o = nn.Linear(head_num * d_v, d_o)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        B, n_q, _ = q.size()
        _, n_k, _ = k.size()
        _, n_v, _ = v.size()
        H = self.head_num

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        Q = q.view(B, n_q, H, self.d_k).transpose(1, 2)  # (B,H,Lq,Dk)
        K = k.view(B, n_k, H, self.d_k).transpose(1, 2)  # (B,H,Lk,Dk)
        V = v.view(B, n_v, H, self.d_v).transpose(1, 2)  # (B,H,Lk,Dv)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B,H,Lq,Lk)

        if mask is not None:
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)  # (B,1,Lq,Lk)
            scores = scores + mask  # 屏蔽处为 -inf

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        head_out = torch.matmul(attn, V)  # (B,H,Lq,Dv)
        head_out = head_out.transpose(1, 2).contiguous().view(B, n_q, H * self.d_v)
        out = self.fc_o(head_out)  # (B,Lq,d_o)
        return attn, out


class FeedForward(nn.Module):
    """前馈网络：两层 MLP，GELU/ReLU 可选。"""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act,
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-LN 编码器层：LN → MHA → 残差，LN → FFN → 残差"""

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        # 按 shousi_MHA 接口构造：dimension_k=dimension_v=embed_dim，d_k=d_v=embed_dim//num_heads，d_o=embed_dim
        self.mha = MHA(num_heads, embed_dim, embed_dim, embed_dim // num_heads, embed_dim // num_heads, embed_dim, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        # 自注意力（MHA 返回 (attn, out)）
        _, out = self.mha(self.ln1(x), self.ln1(x), self.ln1(x), src_mask)
        x = x + self.dropout1(out)
        # 前馈
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)
        return x


class DecoderLayer(nn.Module):
    """Pre-LN 解码器层：
    LN → 自注意力（因果+padding）→ 残差 →
    LN → 交叉注意力（对 encoder 输出）→ 残差 →
    LN → FFN → 残差
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_mha = MHA(num_heads, embed_dim, embed_dim, embed_dim // num_heads, embed_dim // num_heads, embed_dim, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_mha = MHA(num_heads, embed_dim, embed_dim, embed_dim // num_heads, embed_dim // num_heads, embed_dim, dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # 自注意力
        _, out1 = self.self_mha(self.ln1(x), self.ln1(x), self.ln1(x), self_mask)
        x = x + self.dropout1(out1)
        # 交叉注意力（Q=decoder, K/V=encoder memory）
        _, out2 = self.cross_mha(self.ln2(x), self.ln2(memory), self.ln2(memory), cross_mask)
        x = x + self.dropout2(out2)
        # 前馈
        ffn_out = self.ffn(self.ln3(x))
        x = x + self.dropout3(ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ffn_hidden, dropout) for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.pos(self.embed(src))  # (B,L,D)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_hidden, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        trg: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.pos(self.embed(trg))  # (B,L,D)
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    手撕 Transformer（Encoder-Decoder）。
    - mask 约定：传入注意力的 attn_mask 应为 True=需要屏蔽。
    """

    def __init__(
        self,
        src_pad_idx: int,
        trg_pad_idx: int,
        enc_vocab_size: int,
        dec_vocab_size: int,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        n_layers: int,
        max_len: int,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = torch.device(device)

        self.encoder = Encoder(enc_vocab_size, max_len, d_model, ffn_hidden, n_heads, n_layers, dropout)
        self.decoder = Decoder(dec_vocab_size, max_len, d_model, ffn_hidden, n_heads, n_layers, dropout)
        self.generator = nn.Linear(d_model, dec_vocab_size)

    # -------- 掩码构造工具（返回 True=需要屏蔽） --------
    def make_pad_mask(self, q_idx: torch.Tensor, k_idx: torch.Tensor, pad_q: int, pad_k: int) -> torch.Tensor:
        # (B,Lq), (B,Lk) -> (B,1,Lq,Lk)
        q_mask = q_idx.eq(pad_q).unsqueeze(1).unsqueeze(3)  # True=pad 需要屏蔽
        k_mask = k_idx.eq(pad_k).unsqueeze(1).unsqueeze(2)
        return q_mask | k_mask

    def make_causal_mask(self, q_len: int, k_len: int) -> torch.Tensor:
        # (1,1,Lq,Lk) 上三角为 True=需要屏蔽（只允许看自己及之前）
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=self.device).triu(1)
        return causal.unsqueeze(0).unsqueeze(1)

    # -------------------------------------------------

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # src, trg: (B, L)
        B, Ls = src.shape
        _, Lt = trg.shape

        # Encoder 自注意力 mask（仅 padding）
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)  # (B,1,Ls,Ls)

        # Decoder 自注意力 mask（padding | causal）
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)  # (B,1,Lt,Lt)
        trg_causal = self.make_causal_mask(Lt, Lt)  # (1,1,Lt,Lt)
        trg_self_mask = trg_pad_mask | trg_causal

        # 交叉注意力 mask（query=trg，key=src，屏蔽 src 中的 pad）
        cross_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)  # (B,1,Lt,Ls)

        memory = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, memory, trg_self_mask, cross_mask)
        logits = self.generator(dec_out)  # (B,Lt,V)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, Ls, Lt = 2, 10, 9
    Vsrc, Vtrg = 100, 120
    D, H, L, FF = 64, 4, 4, 256

    src_pad_idx = 0
    trg_pad_idx = 0

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_vocab_size=Vsrc,
        dec_vocab_size=Vtrg,
        d_model=D,
        n_heads=H,
        ffn_hidden=FF,
        n_layers=L,
        max_len=128,
        dropout=0.1,
        device=device,
    ).to(device)

    src = torch.randint(0, Vsrc, (B, Ls), device=device)
    trg = torch.randint(0, Vtrg, (B, Lt), device=device)

    # 将部分位置置为 pad 以测试掩码
    src[:, -1] = src_pad_idx
    trg[:, -1] = trg_pad_idx

    logits = model(src, trg)
    print("src:", src.shape, "trg:", trg.shape, "logits:", logits.shape)
```



---
## Transformer框架，不包含编解码器实现
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
```    
## 手撕Transformer完整实现
```python
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦-余弦位置编码（batch_first 版本）。
    输入/输出: (B, L, D)
    """

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        length = x.size(1)
        x = x + self.pe[:, :length, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力（同时支持自注意力与交叉注意力）。
    输入/输出: (B, L, D)，mask 形状: (B, 1, Lq, Lk) 或可广播到此形状。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def _shape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dh)
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dh) -> (B, L, D)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 线性投影
        Q = self._shape_to_heads(self.w_q(q))  # (B,H,Lq,Dh)
        K = self._shape_to_heads(self.w_k(k))  # (B,H,Lk,Dh)
        V = self._shape_to_heads(self.w_v(v))  # (B,H,Lk,Dh)

        # 打分 + 缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Lq,Lk)

        # 掩码：True=需要屏蔽 时，将其置为 -inf（若传入的是 True=可见 的掩码，请在外部取反）
        if attn_mask is not None:
            # 广播到 scores 形状
            while attn_mask.dim() < scores.dim():
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        # softmax → dropout → 与 V 相乘
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights, V)  # (B,H,Lq,Dh)

        # 合并头 + 输出投影
        out = self.out_proj(self._merge_heads(context))  # (B,Lq,D)
        return out, weights


class FeedForward(nn.Module):
    """前馈网络：两层 MLP，GELU/ReLU 可选。"""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act,
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-LN 编码器层：LN → MHA → 残差，LN → FFN → 残差"""

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        # 自注意力
        attn_out, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=src_mask)
        x = x + self.dropout1(attn_out)
        # 前馈
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)
        return x


class DecoderLayer(nn.Module):
    """Pre-LN 解码器层：
    LN → 自注意力（因果+padding）→ 残差 →
    LN → 交叉注意力（对 encoder 输出）→ 残差 →
    LN → FFN → 残差
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # 自注意力
        self_attn_out, _ = self.self_mha(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self_mask)
        x = x + self.dropout1(self_attn_out)
        # 交叉注意力（Q=decoder, K/V=encoder memory）
        cross_attn_out, _ = self.cross_mha(self.ln2(x), self.ln2(memory), self.ln2(memory), attn_mask=cross_mask)
        x = x + self.dropout2(cross_attn_out)
        # 前馈
        ffn_out = self.ffn(self.ln3(x))
        x = x + self.dropout3(ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ffn_hidden, dropout) for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.pos(self.embed(src))  # (B,L,D)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_hidden, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        trg: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.pos(self.embed(trg))  # (B,L,D)
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    手撕 Transformer（Encoder-Decoder）。
    - mask 约定：传入注意力的 attn_mask 应为 True=需要屏蔽。
    """

    def __init__(
        self,
        src_pad_idx: int,
        trg_pad_idx: int,
        enc_vocab_size: int,
        dec_vocab_size: int,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        n_layers: int,
        max_len: int,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = torch.device(device)

        self.encoder = Encoder(enc_vocab_size, max_len, d_model, ffn_hidden, n_heads, n_layers, dropout)
        self.decoder = Decoder(dec_vocab_size, max_len, d_model, ffn_hidden, n_heads, n_layers, dropout)
        self.generator = nn.Linear(d_model, dec_vocab_size)

    # -------- 掩码构造工具（返回 True=需要屏蔽） --------
    def make_pad_mask(self, q_idx: torch.Tensor, k_idx: torch.Tensor, pad_q: int, pad_k: int) -> torch.Tensor:
        # (B,Lq), (B,Lk) -> (B,1,Lq,Lk)
        q_mask = q_idx.eq(pad_q).unsqueeze(1).unsqueeze(3)  # True=pad 需要屏蔽
        k_mask = k_idx.eq(pad_k).unsqueeze(1).unsqueeze(2)
        return q_mask | k_mask

    def make_causal_mask(self, q_len: int, k_len: int) -> torch.Tensor:
        # (1,1,Lq,Lk) 上三角为 True=需要屏蔽（只允许看自己及之前）
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=self.device).triu(1)
        return causal.unsqueeze(0).unsqueeze(1)

    # -------------------------------------------------

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # src, trg: (B, L)
        B, Ls = src.shape
        _, Lt = trg.shape

        # Encoder 自注意力 mask（仅 padding）
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)  # (B,1,Ls,Ls)

        # Decoder 自注意力 mask（padding | causal）
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)  # (B,1,Lt,Lt)
        trg_causal = self.make_causal_mask(Lt, Lt)  # (1,1,Lt,Lt)
        trg_self_mask = trg_pad_mask | trg_causal

        # 交叉注意力 mask（query=trg，key=src，屏蔽 src 中的 pad）
        cross_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)  # (B,1,Lt,Ls)

        memory = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, memory, trg_self_mask, cross_mask)
        logits = self.generator(dec_out)  # (B,Lt,V)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, Ls, Lt = 2, 10, 9
    Vsrc, Vtrg = 100, 120
    D, H, L, FF = 64, 4, 4, 256

    src_pad_idx = 0
    trg_pad_idx = 0

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_vocab_size=Vsrc,
        dec_vocab_size=Vtrg,
        d_model=D,
        n_heads=H,
        ffn_hidden=FF,
        n_layers=L,
        max_len=128,
        dropout=0.1,
        device=device,
    ).to(device)

    src = torch.randint(0, Vsrc, (B, Ls), device=device)
    trg = torch.randint(0, Vtrg, (B, Lt), device=device)

    # 将部分位置置为 pad 以测试掩码
    src[:, -1] = src_pad_idx
    trg[:, -1] = trg_pad_idx

    logits = model(src, trg)
    print("src:", src.shape, "trg:", trg.shape, "logits:", logits.shape)

```
