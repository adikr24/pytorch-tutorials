### a basic introduction to pytorch

import torch as nn
import torch
x= torch.rand(10)
x.size()

output
temp = torch.FloatTensor([10,11,12,13])
temp.size()


temp_II = torch.DoubleTensor([10,11,12,13])
temp_II.size()

##########converting numpy to tensor data
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
boston_tensor = torch.from_numpy(boston.data)

boston_tensor[:2]

### reading an image in python and converting it to 3d tensors
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
os.getcwd()
dir_ = 'C:\\Users\\adity\\Desktop\\ebook'
os.chdir(dir_)


filename = 'DSC_1277.jpg'
with Image.open(filename) as image:
    w,h = image.size

df_img=np.array(Image.open(filename))
df_imgT= torch.from_numpy(df_img)

df_imgT.size()
plt.imshow(df_imgT)

############################################## MAT MUL pytorch
a =  np.array([[2,2],[2,2]])
b =  np.array([[1,1],[1,1]])
a= torch.from_numpy(a)
b= torch.from_numpy(b)
c= a+ b
## different forms of multiplication in pytorch
a*b
a.mul(b)
a.mul_(b)
a.matmul(b)
##################################################
## gradients
from torch.autograd import Variable
x= Variable(torch.ones(2,2), requires_grad=True)
y= x.mean()
print(x.grad)
print(x.grad_fn)
x.data

## creating data for neural network
import torch.nn as nn
## creating a Xlabels and Ylabels
def get_data():
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype= torch.FloatTensor
    X =  Variable(torch.from_numpy(train_X).type(dtype), requires_grad= False).view(17,1)
    y =  Variable(torch.from_numpy(train_Y).type(dtype), requires_grad= False)
    return X,y

# X*w+ b --> Y_pred
## getting additional parameters
def get_weights():
    w= Variable(torch.randn(1), requires_grad = True)
    b = Variable(torch.randn(1), requires_grad= True)
    return w,b

## architecture of the network
def simple_network(x):
    y_pred= torch.matmul(x,w) + b
    return y_pred

## assessing the network and tweaking the weights
def loss_fn(y,y_pred):
    loss = (y-y_pred).pow(2).sum()
    # loss function to be used more than once so all the gradients needs to be cleared before that
   ## first time we call the backward function, the gradients are empty so we zero the gradients only when they are not None
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()    
    return loss.data[0]

## optimizing the network
def optimize(learning_rate):
    w.data= -learning_rate * w.grad.data
    b.data= -learning_rate * b.grad.data
    
##### getting the data in Pytorch format
from torch.utils.data import Dataset, DataLoader
class DogsAndCatsDataset(Dataset):
    ## initialization is done below
    def __init__(self,):
        self.files= glob(root_dir)
        self.size = size
#        pass
    ## __len__returns the maximum number of elements in our dataset
    def __len__(self):
        return len(self.files)
 #       pass
    ## __getitem__ returns an element based on the idx every time it is called
    def __getitem__(self):
        img= np.asarray(Image.open(self.files[idx]).resize(self.size))
        label  = self.files[idx].split('/')[-2]
        return img, label
        #pass

## once the class is created, we can create an object of a class and iterate over it
## getting MNIST dataset from pytorch datasets and exploring it
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import optim

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,),(0.5,)),])
trainset = datasets.MNIST('', download= True, train = True, transform = transform)
trainloader= torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)    

images,label= next(iter(trainloader))
images.size()
im1 = images[0]
im1.size()
im1_plt= np.squeeze(im1)
plt.imshow(im1_plt)




for image,label in trainloader:
    pass
    #Apply your DL on the dataset.
############################################ Linear transformation
from torch.nn import Linear    
## linear layer
l1= Linear(in_features= 10, out_features=5, bias= True)
## inputs
inp = Variable(torch.randn(1,10))
## Apply linear transformation to the inputs
l1(inp).size()
## accessing the trainable parameters

l1.weight ## size of the l1 weight layer would be such that that the mat mul will give out_features so 10X1.T.dot(10X5)--> 
l1.weight.size()
l1.bias

## super is a shortcut to access a base class without having to know its type or name
## here super is used to pass arguments of child class to parents class
## sample network code

class pytorchNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(pytorchNetwork,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)
        
    def __forward__(self,input): 
        out = self.layer1(input)
        out = nn.ReLU(out)
        out = self.layer2(out)
        return out

## 3 types of output layer when coming up with the model architecture
## 1 if the output is continuous then use linear layer with output 1, for predicting continuous vals
## 2 if there is a classification problem then use a sigmoid activation function
## 3 if there is a multi-classification problem then we shall used softmax function
## use cross entropy loss for multi-class problems
def cross_entropy(true_label, prediction):
    if true_label ==1 :
        return -log(prediction)
    else:
        return -log(1-prediction)
##losses
## implementing loss in nn
loss= nn.CrossEntropyLoss()
inp = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.LongTensor(3).random_(5))
output = loss(inp, target)
output.backward()













