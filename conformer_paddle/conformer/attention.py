import paddle
import math
from paddle import nn
from paddle import Tensor
from typing import Optional
import paddle.nn.functional as F
from FFN import Linear


class RelativeMultiHeadAttention(nn.Layer):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.u_bias = self.create_parameter([self.num_heads, self.d_head])
        self.v_bias = self.create_parameter([self.num_heads, self.d_head])

        self.dropout = nn.Dropout(p=dropout_p)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.shape[0]

        query = self.query_proj(query).reshape([batch_size, -1, self.num_heads, self.d_head])
        key = self.key_proj(key).reshape([batch_size, -1, self.num_heads, self.d_head]).transpose([0, 2, 1, 3])
        value = self.value_proj(value).reshape([batch_size, -1, self.num_heads, self.d_head]).transpose([0, 2, 1, 3])
        pos_embedding = self.pos_proj(pos_embedding).reshape([batch_size, -1, self.num_heads, self.d_head])

        content_score = paddle.matmul((query + self.u_bias).transpose([0, 2, 1, 3]), key.transpose([0, 1, 3, 2]))
        pos_score = paddle.matmul((query + self.v_bias).transpose([0, 2, 1, 3]), pos_embedding.transpose([0, 2, 3, 1]))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = paddle.matmul(attn, value).transpose([0, 2, 1, 3])
        context = context.reshape([batch_size, -1, self.d_model])

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.shape
        # zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        zeros = paddle.zeros([batch_size, num_heads, seq_length1, 1])
        padded_pos_score = paddle.concat([zeros, pos_score], axis=-1)

        padded_pos_score = padded_pos_score.reshape([batch_size, num_heads, seq_length2 + 1, seq_length1])
        pos_score = padded_pos_score[:, :, 1:].reshape(pos_score.shape)

        return pos_score


class PositionalEncoding(nn.Layer):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """

    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class MHSA(nn.Layer):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MHSA, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.shape
        pos_embedding1 = self.positional_encoding(seq_length)
        pos_embedding = paddle.concat([pos_embedding1, pos_embedding1], axis=0)
        pos_embedding = paddle.concat([pos_embedding, pos_embedding1], axis=0)
        # pos_embedding = paddle.repeat_interleave(x, 2, 0)
        # pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
