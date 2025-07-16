import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # 1. 断言：确保输出维度可以被头的数量整除
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # 2. 计算每个头的维度
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 4. 定义最后的输出投影层
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # 5. 注册因果掩码????
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 我们用 .view() 将形状为 (b, num_tokens, d_out) 的张量重塑为 (b, num_tokens, num_heads, head_dim)。这步操作没有移动任何数据，只是改变了 PyTorch "看待" 这个张量的方式，巧妙地将最后一个维度拆分成了“头的数量”和“每个头的维度”。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposes from shape 
        # (b, num_tokens, num_heads, head_dim) to 
        # (b, num_heads, num_tokens, head_dim)
        # 交换维度 (.transpose)：这是第二个魔法。我们交换 num_tokens 和 num_heads 的位置。为什么要这么做？因为这样一来，所有头的计算就可以被看作一个更大的批处理操作。PyTorch 的矩阵乘法 @ 会自动在批次维度上（现在是 b * num_heads）并行执行。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # In summary, the transpose(1, 2) operation is not directly performing the parallel computation itself, but it restructures the data in a way that allows PyTorch's efficient batched matrix multiplication to compute the attention scores for all heads simultaneously, rather than sequentially. 

        #  When computing attention scores using the batched matrix multiplication operator (@), PyTorch automatically parallelizes the computation across the leading dimensions. 
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 函数名末尾的下划线 _ 表示这是一个in-place（原地/就地）操作。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec
    
# Sample tensor
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)

# The input embedding size, d=3
d_in = inputs.shape[-1] # Should be the last dimension (features)
# The output embedding size, d=2
d_out = 2

batch = torch.stack((inputs, inputs), dim = 0)
print(batch.shape) # torch.Size([2, 6, 3])

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
