import torch
import torch.nn as nn
from torchviz import make_dot

# 假设我们有一个简单的模型和损失函数
model = nn.Linear(2, 1)
criterion = nn.MSELoss()

# 创建一些随机数据来模拟输入和目标值
inputs = torch.randn(3, 2, requires_grad=True)
targets = torch.randn(3, 1)

# 前向传播
outputs = model(inputs)
loss = criterion(outputs, targets)

# 反向传播
loss.backward()

make_dot(loss).render("CNN_graph")
# 输出每个参数的梯度
for name, param in model.named_parameters():
    print(f"Gradient for {name}: {param.grad}")