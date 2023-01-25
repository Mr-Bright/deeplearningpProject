import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# load and separate data
raw_data = shuffle(pd.read_excel('resale17dd.xlsx'))

train_data = raw_data[0:int(len(raw_data) * 0.9)]
test_data = raw_data[len(train_data):len(raw_data)]
train_x = torch.tensor(train_data.iloc[:, :5].values, dtype=torch.float)
train_y = torch.tensor(train_data.iloc[:, 5].values, dtype=torch.float)[..., None]
test_x = torch.tensor(test_data.iloc[:, :5].values, dtype=torch.float)
test_y = torch.tensor(test_data.iloc[:, 5].values, dtype=torch.float)[..., None]
train_data = Data.TensorDataset(train_x, train_y)
test_data = Data.TensorDataset(test_x, test_y)


# build MLP and 1-d CNN models
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(5, 4)
        self.hidden2 = nn.Linear(4, 10)
        self.output = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.activation(self.output(x))


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()


# set hyper parameters
epochs = 200
lr = 0.01
batch_size = 32
Dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


# train MLP model and test
MLP_model = MLP()
lossMSE = nn.MSELoss()
lossMAE = nn.L1Loss()
opt = optim.Adam(MLP_model.parameters(), lr=lr, betas=(0.5, 0.999))

mse_list = []
mae_list = []
test_mse_list = []
test_mae_list = []
for epoch in range(epochs):
    mse_loss = 0.0
    mae_loss = 0.0
    for batch_x, batch_y in Dataloader:
        opt.zero_grad()
        output = MLP_model(batch_x)
        loss_mse = lossMSE(output, batch_y)
        loss_mae = lossMAE(output, batch_y)
        loss_mse.backward()
        opt.step()
        mse_loss += loss_mse.item()*batch_x.size(0)
        mae_loss += loss_mae.item()*batch_x.size(0)

    print('epoch: {}, train MES: {}, MAS: {}'.format(epoch+1, mse_loss/train_x.size(0), mae_loss/train_x.size(0)))
    mse_list.append(mse_loss/train_x.size(0))
    mae_list.append(mae_loss/train_x.size(0))
    # test mse and mae
    with torch.no_grad():
        test_result = MLP_model(test_x)
        test_loss_mse = lossMSE(test_result, test_y)
        test_loss_mae = lossMAE(test_result, test_y)
        print('epoch: {}, test MES: {}, MAS: {}'.format(epoch+1, test_loss_mse/len(test_y), test_loss_mae/len(test_y)))
        test_mse_list.append(test_loss_mse.item())
        test_mae_list.append(test_loss_mae.item())


x_axis = range(0, epochs)
plt.subplot(2, 2, 1)
plt.plot(x_axis, mse_list, 'o-')
plt.title("train MSE with epochs")
plt.ylabel("MSE")
plt.subplot(2, 2, 2)
plt.plot(x_axis, mae_list, '.-')
plt.title("train MAE with epochs")
plt.ylabel("MAE")
plt.subplot(2, 2, 3)
plt.plot(x_axis, test_mse_list, 'o-r')
plt.title("test MSE with epochs")
plt.ylabel("MSE")
plt.subplot(2, 2, 4)
plt.plot(x_axis, test_mae_list, '.-r')
plt.title("test MAE with epochs")
plt.ylabel("MAE")
plt.show()




