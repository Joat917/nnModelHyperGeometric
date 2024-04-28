"""
训练model1并输出训练完成的模型和训练记录。
"""


from dataManager import *
from model1 import *
import torch.utils.data
import torch.utils.tensorboard
from params import *

data = torch.utils.data.DataLoader(MyDS(), batch_size=64)
loss_func = torch.nn.MSELoss()
model = Model1()
optimizer = torch.optim.Adam(model.parameters(), LR)
writer = torch.utils.tensorboard.writer.SummaryWriter("model1-log")

for epoch in ProgressShower(EPOCH_NUM):
    total_loss = 0
    for x, y in data:
        model.zero_grad()
        optimizer.zero_grad()
        outs = model.forward(x)
        loss = loss_func(outs, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    writer.add_scalar(f"loss-train-dataset{DATASET_NUM}-epoch{EPOCH_NUM}-lr{LR}", total_loss, epoch)

torch.save(model, f"Model1-dataset{DATASET_NUM}-epoch{EPOCH_NUM}-lr{LR}.pth")
writer.close()
