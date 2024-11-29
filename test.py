import torch
from main import test_loader, criterion, DEVICE
from train import model

def test(test_loader, model, loss_fn):

    model.load_state_dict(torch.load("model.pth"))
    model.to(DEVICE)

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()  # 设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():  # 禁用梯度计算
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= num_batches  # 计算平均损失
    accuracy = 100 * correct / size  # 计算准确率
    print(f'Test Results: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n')

test(test_loader, model, criterion)