"""
Created on Tue Mar 12 14:34:11 2024

@author: robertrouse
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from apollo import mechanics as ma
from apollo import metrics as me


### Set plotting style parameters
ma.textstyle()


### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Data import, feature-target identification, and datasplit
df  = pd.read_csv('miniLUSP_output.csv')
df = df.dropna()
columns = df.columns.tolist()
features = columns[1:9]
targets = columns[9:]
xspace = ma.featurelocator(df, features)
yspace = ma.featurelocator(df, targets)
df_train = df.sample(frac=0.8, random_state=42)


### Convert dataframe subsets to arrays and then to PyTorch variables
trnset = df_train.to_numpy()
fullset = df.to_numpy()
X = trnset[:,xspace].reshape(len(trnset), len(xspace)).astype(float)
Y = trnset[:,yspace].reshape(len(trnset), len(yspace)).astype(float)
x = torch.from_numpy(X).to(device)
y = torch.from_numpy(Y).to(device)
x, y = Variable(x), Variable(y)


### Define Neural Network structure and initialisation procedure
class ScapeNET(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ScapeNET, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, out_dim),
            )
    
    def forward(self, z):
        z = self.linear_layers(z)
        return z

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


### Network initialisation
net = ScapeNET(len(xspace), len(yspace))
net = nn.DataParallel(net)
net.apply(init_weights)


### Network training
net = net.train()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0)
loss_func = torch.nn.MSELoss()
loss_list = []
for i in range(16000):
    y_pred = net(x.float())
    loss = loss_func(y_pred, y.float())
    net.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)
    if(i % 500 == 0):
        print('epoch {}, loss {}'.format(i, loss.data))


### Evaluate Network
net = net.eval()
Z = fullset[:,xspace].reshape(len(fullset), len(xspace)).astype(float)
z = torch.from_numpy(Z).to(device)
predicted = net(z.float()).data.cpu().numpy()
df['GWPR_Predicted'] = predicted[:,0]
df['Food_Predicted'] = predicted[:,1]
df['Bird_Predicted'] = predicted[:,2]
df_test = df[~df.isin(df_train)].dropna()
for xf in (df, df_test):
    print('- - - - - - - - - - - - - - -')
    print('R\N{SUPERSCRIPT TWO}: ' + str(me.R2(xf[targets[0]],
                                               xf['GWPR_Predicted'])))
    print('R\N{SUPERSCRIPT TWO}: ' + str(me.R2(xf[targets[1]],
                                               xf['Food_Predicted'])))
    print('R\N{SUPERSCRIPT TWO}: ' + str(me.R2(xf[targets[2]],
                                               xf['Bird_Predicted'])))
torch.save(net, 'model.pt')