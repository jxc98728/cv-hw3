import numpy as np
# load data
from torchvision import datasets
from dataset import MyMNIST, MyCIFAR10
# load the training data
train_data = MyMNIST('./data/mymnist', train=False)
# train_data = MyCIFAR10('./data/mycifar10', train=False)
    
# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
print(len(train_data))
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
# print(x)
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x / 255.0, axis=(0, 1))
train_std = np.std(x / 255.0, axis=(0, 1))
# the the mean and std
print(train_mean, train_std)