import torch.nn as nn
import torch
import math

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads=4, input_size=256, hidden_size=256, hidden_dropout_prob=0):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads

        self.all_head_size = hidden_size #head实际上是在hidden_维度上做切分计算
        self.attention_head_size = self.all_head_size//self.num_attention_heads

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        #x: batch×word_length×hidden_size(all_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # batch×n×num_head×head_size
        x = x.view(*new_x_shape)
        #return x.permute(0, 2, 3, 1)
        return x.permute(0, 2, 1, 3)  # batch×num_head×word_length×head_size

    def forward(self, input_tensor):
        # 16×2×256
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # 16×4×2×64 多头拆分
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 16×4×2×2
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   # batch×num_head×1×1

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # 16×4×2×2
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        # 16×4×2×64
        context_layer = torch.matmul(attention_probs, value_layer)

        # 16×2×4×64
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #16×2×256 分头结果复原
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # 16×2×256
        #hidden_states=nn.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

num_attention_heads=4
input_tensor=torch.rand(16,8,256)  # batch,feature_num,feature
multihead_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0)
model=SelfAttention(num_attention_heads,256,256)
out=model(input_tensor)
attn_output = multihead_attn(query=input_tensor, key=input_tensor, value=input_tensor)
print(out.size())
print(attn_output[0].shape)
