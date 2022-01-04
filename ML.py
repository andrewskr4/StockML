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
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5,2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

def main():

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model = Net().to(device)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    raw_data = np.load('GIS_train_set.npy')
    x=np.array([i[0:-1] for i in raw_data])
    y=np.array([])
    for i in raw_data:
        if len(y)==0:
            if(i[-1]==1):
                y=np.append(y, [1,0])
            else:
                y=np.append(y, [0,1])
        else:
            if(i[-1]==1):
                y=np.vstack([y, [1,0]])
            else:
                y=np.vstack([y, [0,1]])

    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(dataset,batch_size=100)


    for epoch in range(1, 50):
        train(model, device, train_loader, optimizer, epoch, criterion)
        #test(model, device, test_loader)

    
    torch.save(model.state_dict(),"GeneralMillsMLModel.pt")
       
if __name__ == '__main__':
    main()