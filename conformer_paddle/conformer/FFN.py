from paddle import nn
from paddle import Tensor


# modules
class Linear(nn.Layer):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        w_tmp = self.create_parameter([in_features, out_features], is_bias=bias)
        self.add_parameter("w_tmp", w_tmp)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class Swish(nn.Layer):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.sigmoid(inputs)


class GLU(nn.Layer):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, axis=self.dim)
        return outputs * gate.sigmoid()


# sub_block
class FFN(nn.Layer):
    def __init__(self,
                 encoder_dim: int = 512,
                 expansion_factor: int = 4,
                 dropout_p: float = 0.1,
                 ) -> None:
        super(FFN, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
