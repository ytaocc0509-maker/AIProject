import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional



@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32  # transformer block 堆叠数量
    n_heads: int = 32  # heads 中 Q 的数量, 回顾 MQA GQA
    n_kv_heads: Optional[int] = None  # heads中 K 和 V 的数量
    vocab_size: int = -1  # 这个值在加载分词器设置
    multiple_of: int = 256  # FFN 网络中隐藏神经元数量
    ffn_dim_multiplier: Optional[float] = None  # 当使用GQA之后, K和V的数量会减少, 但是增加FFN中神经元数量
    norm_eps: float = 1e-5

    # 参数给 KV cache 所用
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # 预先计算 RoPE 中需要的 mθ
    assert head_dim % 2 == 0, "必须可以被2整除,因为公式中 d/2"

    # 构建 theta 参数
    # 根据论文中的公式实现
    theta_numerator = torch.arange(0, head_dim, 2).float()  # 2(i-1)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))  # 10000^(-2(i-1)/d)

    # 构建 m 参数, 代表着 positions 位置
    m = torch.arange(seq_len, device=device)

    # 接下来 mθ 两个序列内积, 这里我们要得到所有的排列组合, 用 torch.outer
    # 这样每个 position 都有一组 mθ 值
    freqs = torch.outer(m, theta).float()

    # 我们可以用极坐标形式计算复数
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 1, 将 x token 向量中的 dimension 个值分组, 2个值一组
    # 2, 然后将其转换为复数形式
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # 3, 乘上我们准备好的矩阵
    x_rotated = x_complex * freqs_complex

    # 4, 将复数 a+ib 形式中的 a 和 b 提取出来
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 公式中的 g 参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim)
        # torch.rsqrt() 简单来说就是对每个元素开根号后再取倒数
        # -1 是对最后一个维度求平均
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self._norm(x.float()).type_as(x) * self.weight
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "必须设定词表大小"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # input embedding
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            # EncoderBlock 是之后要去实现的 LLaMA transformer block
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # 稍后会展开去讲, 现在先写在这里
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # 这里实现的是 inference, 所以每次传入一个token, 那么 seq_len 一直都是 1
        # tokens 的形状是 (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "每次只处理一个token"

        # (B, seq_len) -> (B, seq_len, Dim)
        h = self.tok_embeddings(tokens)
        # 先去获取对应的 positional encoding 相关信息
        # 根据位置 [start_pos, start_pos+seq_len] 获取 (m, theta)
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # 连续应用 encoder layers / transformer block
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output

