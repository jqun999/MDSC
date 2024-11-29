#定义训练方法
import torch
from torch.cuda import device
from main import train_loader, model, criterion, optimizer, EPOCHS, DEVICE


def train(train_loader, model, loss_fn, optimizer, epochs, DEVICE):
    size = len(train_loader.dataset)
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(train_loader):  # (data,target)
            X, y = X.to(DEVICE), y.to(DEVICE)
            # 计算预测和损失
            pred = model(X)
            loss = loss_fn(pred, y)

            # 反向传播和优化
            optimizer.zero_grad()  # 梯度初始化为0
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch:>3d}], Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

train(train_loader, model, criterion, optimizer, EPOCHS, DEVICE)