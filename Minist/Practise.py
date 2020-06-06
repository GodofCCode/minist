#  https://www.cnblogs.com/wj-1314/p/9842719.html     代码练习网址

#_*_coding:utf-8_*_
import matplotlib.pyplot as plt
import numpy as np
import CV2
import time
import torch
# torchvision包的主要功能是实现数据的处理，导入和预览等
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autogard import Variable

start_time = time.time()
# 对数据进行载入及有相应变换,将Compose看成一种容器，他能对多种数据变换进行组合
# 传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
data_train = datasets.MNIST(root='./data',
                            transforms=transform)

