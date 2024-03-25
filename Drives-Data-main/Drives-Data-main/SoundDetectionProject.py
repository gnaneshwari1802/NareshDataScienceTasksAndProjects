# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:55:57 2023

@author: M GNANESHWARI
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# 라이브러리 불러오기
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import torchaudio
# =============================================================================
# import torchaudio
# C:\Users\M GNANESHWARI\Anaconda3\envs\tensorflow_cpu\lib\site-packages\torchaudio\backend\utils.py:67: UserWarning: No audio backend is available.
#   warnings.warn('No audio backend is available.')
# 
# !pip install soundfile
#Collecting soundfile
# =============================================================================

import pandas as pd
import torch
import torch.nn as nn

class MonoToColor(nn.Module):
    def __init__(self, num_channels=3):
        super(MonoToColor, self).__init__()
        self.num_channels = num_channels

    def forward(self, tensor):
        return tensor.repeat(self.num_channels, 1, 1)
##IDK what is wrong with that up cell
import torch
import subprocess


#The subprocess module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes. This module is commonly used for tasks like running system commands, executing shell scripts, and interacting with the command line from within your Python code.

def GPU():
    if torch.cuda.is_available() == True:
        #The torch.cuda.is_available() function is used to check whether a CUDA-compatible GPU (Graphics Processing Unit) is available and accessible for use with PyTorch. CUDA is a parallel computing platform and API developed by NVIDIA that allows you to leverage the power of GPUs for accelerating deep learning computations.
        device = 'cuda'
        templist = [1, 2, 3]
        templist = torch.FloatTensor(templist).to(device)
        #torch.FloatTensor is a class in PyTorch that represents a multi-dimensional tensor with floating-point values. Tensors are the fundamental data structures in PyTorch and are used for various operations in deep learning, including storing data, performing mathematical operations, and training neural networks.
        print("Cuda torch working : ", end="")
        print(templist.is_cuda)
        print("current device no. : ", end="")
        print(torch.cuda.current_device())
        print("GPU device count : ", end="")
        print(torch.cuda.device_count())
        print("GPU name : ", end="")
        print(torch.cuda.get_device_name(0))
        print("device : ", device)
        # Execute the nvidia-smi command using subprocess
        try:
            output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            print("nvidia-smi output:")
            print(output)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("Error executing nvidia-smi command:", str(e))
    
    else:
        print("cant use gpu , activating cpu")
        device = 'cpu'

    return device
device = GPU()
print(device) 
# Dataset 클래스 정의
class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        self.annotations = (annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.target_sample_rate:
            signal = signal[:, :self.target_sample_rate]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.target_sample_rate:
            num_missing_samples = self.target_sample_rate - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        file_name = self.annotations.iloc[index, 0]
        audio_sample_path = os.path.join(self.audio_dir, fold, file_name)
        return audio_sample_path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]   
    # 데이터셋 및 데이터 로더 설정
import pandas as pd
import requests
from io import StringIO
# Load the dataset
#url='https://www.kaggle.com/code/cafalena/urban-8k-deeplearning-pytorch/input?select=UrbanSound8K.csv'
# =============================================================================
# response = requests.get(url)
# data = StringIO(response.text)
# =============================================================================
ANNOTATIONS_FILE = pd.read_csv("C:/Users/M GNANESHWARI/Downloads/UrbanSound8K.csv",header=None)







AUDIO_DIR = "C:/Users/M GNANESHWARI/Downloads"

# =============================================================================
# =============================================================================
# # # =============================================================================
# # # 
# # # =============================================================================
# =============================================================================
# =============================================================================
SAMPLE_RATE = 22050
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True if torch.cuda.is_available() else False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

transformation = transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128),
    torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80),
    MonoToColor()
])


usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, transformation, SAMPLE_RATE)

# 데이터셋 분리
dataset_size = len(usd)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(usd, [train_size, val_size, test_size])

# 데이터 로더 생성
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

# ResNet18 모델 설정
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)  # UrbanSound8K의 클래스 개수인 10으로 변경
# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model = model.to(device)

# 손실함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러 설정
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
from tqdm import tqdm
# 최고 검증 정확도를 저장하기 위한 변수 설정
best_acc = 0.0

# 모델 훈련 함수 정의
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    global best_acc

    for epoch in range(num_epochs):
        if epoch % 20 == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # 각 에포크(epoch)은 학습 단계와 검증 단계를 거칩니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in tqdm(dataloaders[phase]):
                #print(input.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계를 계산
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if epoch % 20 == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0:
            print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

import copy
NB_EPOCH = 100
# 모델 훈련 시작
dataloaders = {"train": train_loader, "val": val_loader}
# define dataset_sizes
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NB_EPOCH)

# 가장 좋은 모델 저장
torch.save(best_model.state_dict(), "ResNet18_Best.pth")

# 모델 평가 함수 정의
def test_model(model, test_loader, device):
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on test images: {100 * correct / total}%')

# 훈련 및 평가
test_model(best_model, test_loader, device)
