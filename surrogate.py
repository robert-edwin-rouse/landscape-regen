#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:19:05 2025

@author: robertrouse
"""

import torch.nn as nn 
import torch.optim as tt

### Define Neural Network structure and initialisation procedure
class LandNET(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LandNET, self).__init__()
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

def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def training(m, x, y, device, epochs=16000, opt=tt.Adam, lr=0.0005, decay=0,
             reporting_interval=500):
    m = m.train()
    m = m.to(device)
    optimizer = opt(m.parameters(), lr=lr, weight_decay=decay)
    loss_func = nn.MSELoss()
    loss_list = []
    for i in range(epochs):
        y_pred = m(x.float())
        loss = loss_func(y_pred, y.float())
        m.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
        if(i % reporting_interval == 0):
            print('epoch {}, loss {}'.format(i, loss.data))
    m.eval()