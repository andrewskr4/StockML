from os import X_OK
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Make sure input to first layer is the same as sample_length in prep_data.py
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10,2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.log_softmax(x, dim=1)
        return x
   
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    cum_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        cum_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(cum_loss/len(train_loader))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #output = np.round(output)
            #print(output)
            if (output[0][0]>output[0][1]):
                output[0] = torch.Tensor([1,0])
            elif (output[0][0]<output[0][1]):
                output[0] = torch.Tensor([0,1])
            else:
                output[0] = torch.Tensor([0,1])
            #print(output, target)
            if (torch.equal(output[0], target[0])):
                correct +=1
    print("Percent correct: "+str(correct/len(test_loader) * 100)+"%")

def main():

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model = Net().to(device)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.08)

    raw_data = np.load('GIS_train_set.npy')
    train_raw = np.array([])
    for i in range(len(raw_data)-500):
        if (i==0):
            train_raw = np.append(train_raw, raw_data[i])
        else:
            train_raw = np.vstack([train_raw, raw_data[i]])
    print(len(train_raw))

    test_raw = np.array([])
    for i in range(len(raw_data)-500, len(raw_data)):
        if (i==len(raw_data)-500):
            test_raw = np.append(test_raw, raw_data[i])
        else:
            test_raw = np.vstack([test_raw, raw_data[i]])
    print(len(test_raw))

    x_train=np.array([i[0:-1] for i in train_raw])
    y_train=np.array([])
    for i in train_raw:
        if len(y_train)==0:
            if(i[-1]==1):
                y_train=np.append(y_train, [1,0])
            else:
                y_train=np.append(y_train, [0,1])
        else:
            if(i[-1]==1):
                y_train=np.vstack([y_train, [1,0]])
            else:
                y_train=np.vstack([y_train, [0,1]])

    x_test=np.array([i[0:-1] for i in test_raw])
    y_test=np.array([])
    for i in test_raw:
        if len(y_test)==0:
            if(i[-1]==1):
                y_test=np.append(y_test, [1,0])
            else:
                y_test=np.append(y_test, [0,1])
        else:
            if(i[-1]==1):
                y_test=np.vstack([y_test, [1,0]])
            else:
                y_test=np.vstack([y_test, [0,1]])

    train_tensor_x = torch.Tensor(x_train)
    train_tensor_y = torch.Tensor(y_train)

    test_tensor_x = torch.Tensor(x_test)
    test_tensor_y = torch.Tensor(y_test)

    
    dataset = TensorDataset(train_tensor_x, train_tensor_y)
    testset = TensorDataset(test_tensor_x, test_tensor_y)
    train_loader = DataLoader(dataset,batch_size=100)
    test_loader = DataLoader(testset)


    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch, criterion)
    test(model, device, test_loader)

    
    torch.save(model.state_dict(),"GeneralMillsMLModel.pt")
       
if __name__ == '__main__':
    main()