import torch
import os,glob
import random,csv
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import visdom
import time

class Pokeman(Dataset):
    """
    自定义数据集操作的类，继承自Dataset，需要实现这两个方法
    """
    def __init__(self,root,resize,mode):
        """
        初始化时接受3个参数
        :param root: 这是图片保存的路径
        :param resize: 初始化图片的固定大小
        :param mode: 使用字符串标识时train，val，test
        """
        super(Pokeman, self).__init__()
        self.root = root
        self.resize = resize

        # 将各个类别对应相应的标签
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue

            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        self.images,self.labels = self.load_csv('images.csv')

        # 以6：2：2的比例划分训练集，验证集，测试集
        if mode=='train':
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode=='val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root,name,'*.png'))
                images += glob.glob(os.path.join(self.root,name,'*.jpg'))
                images += glob.glob(os.path.join(self.root,name,'*.jpeg'))

            # print(len(images),images)

            # 先打乱图片
            random.shuffle(images)

            # 将每张图片的路径以及对应的标签写入到csv文件中
            with open(os.path.join(self.root,filename),mode='w',newline='')as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img,label])

            print('以写入csv文件',filename)

        # 如果csv文件已经存在，就将csv中的内容读取出来
        images,labels = [],[]
        with open(os.path.join(self.root,filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img,label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) ==len(labels)

        # print(images,labels)
        return images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        将图片从路径中获取
        :param idx: 图片的索引
        :return:返回tensor格式的img和label
        """
        img,label = self.images[idx],self.labels[idx]

        # 将图片路径转换为image data
        tf = transforms.Compose([
            lambda x : Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            # 做一些简单的数据增强操作
            transforms.RandomRotation(15) ,# 这里旋转的角度不建议太大
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img,label

    def Denormalize(self,x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat*std+mean
        return x

def main():

    viz = visdom.Visdom()

    db = Pokeman('./Data/pokeman',224,'training')
    x,y = next(iter(db))
    print(x.shape,y)
    viz.image(db.Denormalize(x),win='一个样本',opts=dict(title = '样本x'))
    # viz.image(x,win='一个样本',opts=dict(title = '样本x'))

    # 加载一个batch的图片，并打乱
    data_loader = DataLoader(db,batch_size=32,shuffle=True)
    for x,y in data_loader:
        viz.images(db.Denormalize(x),nrow=8,win='batch',opts=dict(title = 'batch'))
        # viz.text(str(y.numpy()),win='batch',opts=dict(title = 'batch_y'))

        time.sleep(10)

if __name__ == '__main__':
    main()