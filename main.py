# created by yan-x-p 2019.04.18
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
from dataset import *
from torchvision import transforms
from  tensorboardX import SummaryWriter

chars = {"深": 0, "秦": 1, "京": 2, "海": 3, "成": 4, "南": 5, "杭": 6, "苏": 7, "松": 8, "0": 9, "1": 10, "2": 11,
                 "3": 12, "4": 13, "5": 14,
                 "6": 15, "7": 16, "8": 17, "9": 18, "A": 19, "B": 20, "C": 21, "D": 22, "E": 23, "F": 24, "G": 25,
                 "H": 26,
                 "J": 27, "K": 28, "L": 29, "M": 30, "N": 31, "P": 32, "Q": 33, "R": 34, "S": 35, "T": 36, "U": 37,
                 "V": 38,
                 "W": 39, "X": 40, "Y": 41, "Z": 42}

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--training",action='store_true',help='whether training')
parser.add_argument('-g', "--use_gpu",action='store_true',help='whether using gpu')
parser.add_argument("-b", "--batchsize", default=1024,help="batch size for train")
parser.add_argument("-n", "--epochs", default=100,help="the epochs for train",type=int)
parser.add_argument("-c", "--load_ckpt", help="checkpoint path to load for prediction",default='best_model.pth')
parser.add_argument('-r', "--resume",action='store_true',help='whether load checkpoint for finetuning')
parser.add_argument("-i", "--images",help="the image dir",default='data')
parser.add_argument("-train", "--train_path",help="the file of train ",default='generate-data-label.txt')
parser.add_argument("-test", "--test_path",help="the file of test",default='test-label.txt')
args = parser.parse_args()

def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

class CarNet(nn.Module):
    def __init__(self,numclasses):
        super(CarNet, self).__init__()
        h1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        h2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        h3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        h4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.feature = nn.Sequential(h1,h2,h3,h4)
        self.dropout = nn.Dropout(0.2)
        self.classifier0 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier4 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier5 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier6 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier7 = nn.Sequential(
            nn.Linear(53248, numclasses))
        self.classifier8 = nn.Sequential(
            nn.Linear(53248, numclasses))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        y0 = self.classifier0(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        y3 = self.classifier3(x)
        y4 = self.classifier4(x)
        y5 = self.classifier5(x)
        y6 = self.classifier6(x)
        y7 = self.classifier7(x)
        y8 = self.classifier8(x)
        return [y0, y1, y2, y3, y4, y5, y6,y7,y8]

def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(len(labelGT))]
    return sum(compare)

def eval_model(model):
    im_aug = transforms.Compose([
        transforms.Resize((70,356)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    correct,count= 0,0
    testdata = CarDataLoader(args.images,args.test_path,chars,transform=im_aug)
    testloader = DataLoader(testdata, batch_size=1, shuffle=True, num_workers=8)
    for i, (img,labelGT) in enumerate(testloader):
        if args.use_gpu:
            x = Variable(img.cuda())
        else:
            x = Variable(img)
        y_pred = model(x)
        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]
        correct += isEqual(labelGT,labelPred)
        count += len(labelGT)
    return correct,count


def pred_model(model_path):
    model = CarNet(len(chars))
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.load_state_dict(torch.load(model_path))
    if args.use_gpu:
        model = model.cuda()
    model.eval()
    im_aug = transforms.Compose([
        transforms.Resize((70,356)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    testdata = CarTestDataLoader(args.images,args.test_path, transform=im_aug)
    testloader = DataLoader(testdata, batch_size=1, shuffle=True, num_workers=8)
    fs = open('pred-data.txt','w')
    for i, (img,img_name) in enumerate(testloader):
        label = ''
        if args.use_gpu:
            img = img.cuda()
        x = Variable(img)
        y_pred = model(x)
        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]
        for j in range(len(labelPred)):
            if labelPred[j]<9:
                label += str(labelPred[j])
            else:
                label += get_key(chars,labelPred[j])[0]
        obj = label + ',  ' + img_name[0] + '\r\n'
        fs.write(obj)
    fs.close()

def train_model():
    im_aug = transforms.Compose([
        transforms.Resize((70, 356)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model = CarNet(len(chars))
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if args.resume:
        model.load_state_dict(torch.load(args.load_ckpt))
    if args.use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    traindata = CarDataLoader(args.images,args.train_path, chars, transform=im_aug)
    trainloader = DataLoader(traindata, batch_size=args.batchsize, shuffle=True, num_workers=8)
    best_acc = 0
    writer = SummaryWriter(log_dir='logs')
    for epoch in range(0, args.epochs):
        lossAver = []
        model.train(True)
        for i, (imgs,labels) in enumerate(trainloader):
            loss = 0.0
            if args.use_gpu:
                x = Variable(imgs.cuda())
            else:
                x = Variable(imgs)
            y_pred = model(x)
            for j in range(len(labels)):
                if args.use_gpu:
                    l = Variable(labels[j]).cuda()
                else:
                    l = Variable(labels[j])
                loss += criterion(y_pred[j], l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossAver.append(loss.data[0])
        model.eval()
        correct,count = eval_model(model)
        acc = correct/count
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
        writer.add_scalar('logs/loss',loss,epoch)
        writer.add_scalar('logs/acc',acc,epoch)
        print('epoch:%s,acc:%.2f%%,loss:%.2f'%(epoch,acc*100,sum(lossAver)/len(lossAver)))
    writer.close()

if __name__ == '__main__':

    if args.training:
        train_model()
    else:
        pred_model(args.load_ckpt)