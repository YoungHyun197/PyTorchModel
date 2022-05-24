## 라이브러리 추가
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import UNet
from dataset import *
from utils import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", deafault = 4 , type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="./train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## 트레이닝 파라미터 설정하기
lr =args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

# gpu를 사용할지 cpu를 사용할지 결정하는 flag
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기 (데이터 불러오기)

if mode == 'train':

    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    # tranining dataset
    dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    # validation dataset
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform = transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # 그 밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    ## 데이터 불러오기
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    # tranining dataset
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # 그 밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(),lr=lr)

## 그 밖에 부수적인 functnions 설정하기
# tensor에서 numpy로 transpose 시키는 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# normalization 되어 있는 data를 반대로 denormalization하는 함수
fn_denorm = lambda x, mean, std : (x*std) + mean
# network output을 binary class로 분류해주는 함수
fn_class = lambda x: 1.0 * (x>0.5)

## Tensorboard를 사용하기 위한 summarywriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir, 'val'))


## 네트워크 학습시키기 (for loop)
st_epoch = 0

if mode == 'train':
    if train_continue =="on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch +1, num_epoch +1):
        net.train() # network 에게 training임을 알려주는 함수 활성화
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # network에게 입력을 받아 ouput을 출력하는  하는 forward path
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward path
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("Train: Epoch %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorbaord 저장하기 (input, output, label)
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch-1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('ouput', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        # loss를 tensorboard에 작성
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        #network validtation (backward 없음 )
        with torch.no_grad():
            net.eval() # eval 모드 명시
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print('VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f' %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('ouput', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
        #loss function 저장
        writer_val.add_scalar(' loss', np.mean(loss_arr), epoch)

        #epoch 5번 진행마다 네트워크를 저장
        if epoch % 5 ==0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    #학습이 완료되면 writer 닫아주기
    writer_train.close()
    writer_val.close()
#TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # network validtation (backward 없음 )
    with torch.no_grad():
        net.eval()  # eval 모드 명시
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print('TEST: | BATCH %04d / %04d | LOSS %.4f' %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            # for loop을 추가하여 각각의 슬라이스를 따로 따로 저장
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                # png file로 저장하기
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # numpy type으로 저장하기
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))
##

