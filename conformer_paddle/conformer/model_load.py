import paddle
import paddle.nn as nn
from model import Conformer

batch_size, sequence_length, dim = 3, 12341, 80

print(paddle.device.get_device())
device = paddle.device.get_device()

criterion = nn.CTCLoss()

inputs = paddle.rand([batch_size, sequence_length, dim])
input_lengths = paddle.to_tensor([12345, 12300, 12000], dtype='int64')
targets = paddle.to_tensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]], dtype='int32')
target_lengths = paddle.to_tensor([9, 8, 7], dtype='int64')

model = Conformer(num_classes=10,
                  input_dim=dim,
                  encoder_dim=32,
                  num_encoder_layers=3)

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

# Calculate CTC Loss
print(outputs.shape)
print(outputs.dtype, targets.dtype, output_lengths.dtype, target_lengths.dtype)
loss = criterion(outputs.transpose([1,0,2]), targets, output_lengths, target_lengths)
print(loss)
