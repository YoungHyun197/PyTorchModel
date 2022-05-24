import os
import numpy as np
import torch
import torch.nn as nn
## 데이터 로더 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # self.data_dir에 있는 모든 파일들을 list_data에 리스트로 저장

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input
    # 함수의 length를 확인
    def __len__(self):
        return len(self.lst_label)
    # 실제로 데이터를 불러오는 함수 (index에 해당하는 파일을 불러옴)
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])).astype(np.float32)
        input = np.load(os.path.join(self.data_dir, self.lst_input[index])).astype(np.float32)

        #0~1 사이의 값을 갖도록 하기위해 255로 나누어 normalize 해준다.
        label = label / 255.0
        input = input / 255.0

        # x, y, channel axis등 총 3개의 axis필요
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        #만약 인자로 transform이 정의되어 있다면 transform을 통과한 dataset을 return 하도록 구현
        if self.transform:
            data = self.transform(data)

        return data



## 트랜스폼 구현
class ToTensor(object):
    def __call__(self, data):
        label = data['label']
        input = data['input']
        # numpy (x, y, ch) -> pytorch (ch, x, y)로 변경
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input':torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label = data['label']
        input = data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label = data['label']
        input = data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
