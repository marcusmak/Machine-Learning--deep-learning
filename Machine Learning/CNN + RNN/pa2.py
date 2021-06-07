# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Assignment 2
# ## Part 1
# 
# 
# %% [markdown]
# #### 4.1

# %%
from skimage import io , transform
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
part1_path = './patched_pa2_data/pa2_data 2/part1_data/'
# sub_set = pd.read_csv("./patched_pa2_data/pa2_data 2/part1_data/submission.csv")
# test_set = pd.read_csv(part1_path + "test.csv")
train_df = pd.read_csv(part1_path + "train.csv")
print(train_df.head())
class posterDataset(Dataset):

    def __init__(self, root_path, dataFrame, transform = None):
        # self.file_path = file_path
        # self.classes = [0, 1]
        self.root_path = root_path
        self.transform = transform
        # Some preprocessing
   
        self.df = dataFrame

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_path + 'images/' + str(self.df.iloc[idx,0]) + '.jpg'
        X = io.imread(img_name)
        y = np.asarray(self.df.iloc[idx, 1:], dtype = np.float32)
        sample = {"image" : X , "genre" : y}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        if(sample['image'].shape[0] == 1):
            print(self.df.iloc[idx,:])
        return sample

    def __len__(self):
        return len(self.df)

# img_names = file_name(train_set['imdbId'].apply(str))
# plt.imshow(io.imread(part1_path + 'images/118589.jpg'))
# io.imread(part1_path + 'images/118589.jpg')


# %%
from torchvision import transforms
transform = transforms.Compose(
    [transforms.ToTensor(),    # range [0, 255] -> [0.0,1.0]
     transforms.Normalize((0.5, ), (0.5, ))])  
full_set = posterDataset(part1_path,train_df,transform) 


# %%
from torch.utils.data import random_split

train_size = int(0.8 * len(full_set))
train_set, val_set = random_split(full_set, [train_size, len(full_set)-train_size ])
# _,_,train_set, val_set = random_split(full_set, [train_size, len(full_set)-train_size-110, 100 ,10 ]
train_set[10]['image'].shape[0]


# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #268 x 182 x 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  #264 x 178 x 6
        self.pool = nn.MaxPool2d(2, 2) #132 x 89 x 6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10,  kernel_size=7)  #126 x 83 x 10
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=16,  kernel_size=11)  #53 x 31 x 16
        self.fc1 = nn.Linear(6240, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        #  3, 32, 32
        # out_dim = in_dim - kernel_size + 1  
        x = self.pool(F.relu(self.conv1(x))) #6, 14, 14 
        x = self.pool(F.relu(self.conv2(x))) #16, 5, 5
        x = self.pool(F.relu(self.conv3(x))) #16, 5, 5
        x = x.view(-1, 6240)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
from torchsummary import summary
model = Net()
summary(model, input_size=(3,268, 182))


# %%
import traceback
import logging
def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')

def load_checkpoint(model, optimizer):
    save_path = f'cifar_net.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')
    
    return val_loss



def TRAIN(net, train_loader, valid_loader,  num_epochs, eval_every, total_step, criterion, optimizer, val_loss, device, save_name):
    loss_his = []
    running_loss = 0.0
    # running_corrects = 0
    # running_num = 0
    global_step = 0
    if val_loss == None:
        best_val_loss = float("Inf")  
    else: 
        best_val_loss = val_loss
    

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, samples in enumerate(train_loader):
            inputs = samples['image']
            labels = samples['genre']
            net.train()                             # set to trainning mode
            inputs = inputs.to(device)              
            labels = labels.to(device)
            '''Training of the model'''
            # Forward pass
            outputs = net(inputs)
            # _, preds = torch.max(outputs.data, 1)   # get the predicted output val,index


            loss = criterion(outputs, labels)       # calc the loss
            
            # Backward and optimize
            optimizer.zero_grad()                   
            loss.backward()                         # calc grad
            optimizer.step()                        # refreash the weight
            global_step += 1

            running_loss += loss.item()
            # running_corrects += torch.sum(preds == labels.data)
            # running_num += len(labels)
            
            '''Evaluating the model every x steps'''
            if global_step % eval_every == 0:
                with torch.no_grad():
                    net.eval()
                    val_running_loss = 0
                    
                    # val_running_corrects = 0
                    try:
                        for samples in valid_loader:
                            val_inputs = samples['image']
                            val_labels = samples['genre']
                            val_outputs = net(val_inputs)
                            val_loss = criterion(val_outputs, val_labels)
                            # _, preds = torch.max(val_outputs.data, 1)
                            val_running_loss += val_loss.item()
                            # val_running_corrects += torch.sum(preds == val_labels.data)
                    except Exception as e:
                        print(global_step)
                        return valid_loader
                        
                    average_train_loss = running_loss / eval_every
                    average_val_loss = val_running_loss / len(valid_loader)
                    # average_train_acc = running_corrects / float(running_num)
                    # average_val_acc = val_running_corrects / float(len(valid_loader))

                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, num_epochs, global_step, total_step, average_train_loss,average_val_loss))

                    running_loss = 0.0
                    running_num = 0
                    # running_corrects = 0
                    
                    if average_val_loss < best_val_loss:
                        best_val_loss = average_val_loss
                        save_checkpoint(save_name, net, optimizer, best_val_loss)
                    
                    

    print('Finished Training')


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# %%
num_epochs = 1
eval_every = 1000
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
                                          shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=10,
                                          shuffle=True)

total_step = len(train_loader)*num_epochs
best_val_loss = None
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters() , lr=0.001)
save_path = f'cifar_net.pt'
model = model.to(device)


catcher = TRAIN(model, train_loader, valid_loader, num_epochs, eval_every, total_step, criterion, optimizer, best_val_loss, device, save_path)


# %%
catcher


# %%


