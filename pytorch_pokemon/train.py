import torch
from torch import optim,nn
import visdom
import torchvision
from pokemon import Pokeman
from torch.utils.data import  DataLoader
from resnet import ResNet18


# 一些参数设置
batch_size = 32
lr = 1e-3
epochs = 20
device = torch.device('cuda')
torch.manual_seed(1234)

# 加载数据集
train_db = Pokeman('./Data/pokeman',224,'training')
val_db = Pokeman('./Data/pokeman',224,'val')
test_db = Pokeman('./Data/pokeman',224,'test')

train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,num_workers=4)
val_loader = DataLoader(val_db,batch_size=batch_size,num_workers=2)
test_loader = DataLoader(test_db,batch_size=batch_size,num_workers=2)

# 定义验证和测试函数
def evaluate(model,loader):
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            predict = logits.argmax(dim=1)

        correct += torch.eq(predict,y).sum().float().item()

    return correct/total


def main():
    model = ResNet18(5).to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc,best_epoch = 0,0

    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            logits = model(x)
            loss = criterion(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 ==0 :
            val_acc = evaluate(model,val_loader)
            if val_acc>best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(),'best.model')

    print('最高准确率',best_acc,'当时的epoch',best_epoch)
    model.load_state_dict(torch.load('best.model'))
    print('加载验证集上最好的模型，进行测试')
    test_acc = evaluate(model,test_loader)
    print('测试准确率',test_acc)

if __name__ == '__main__':
    main()
