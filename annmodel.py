"""
Created on Tue Mar 12 14:34:11 2024

@author: robertrouse
"""

import pandas as pd
import torch
import torch.nn as nn
import surrogate as sr
from torch.autograd import Variable
from apollo import mechanics as ma
from apollo import metrics as me


def main(overwrite=False):
    ### Set reproducibility parameters and devices
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Data import, feature-target identification, and datasplit
    df = pd.read_csv('miniLUSP_output.csv').dropna()
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
    x = Variable(torch.from_numpy(X).to(device))
    y = Variable(torch.from_numpy(Y).to(device))
    
    ### Network initialisation
    net = sr.LandNET(len(xspace), len(yspace))
    net = nn.DataParallel(net)
    net.apply(sr.init_weights)
    
    ### Network training & evaluation
    sr.training(net, x, y, device)
    Z = fullset[:,xspace].reshape(len(fullset), len(xspace)).astype(float)
    z = torch.from_numpy(Z).to(device)
    predicted = net(z.float()).data.cpu().numpy()
    prediction_names = ['GWPR_Predicted', 'Food_Predicted', 'Bird_Predicted']
    df = pd.concat([df, pd.DataFrame(predicted, columns=prediction_names)], axis=1)
    df_test = df[~df.isin(df_train)].dropna()
    r2_string = 'R\N{SUPERSCRIPT TWO}: '
    for xf in (df, df_test):
        print('- - - - - - - - - - - - - - -')
        print(r2_string + str(me.R2(xf[targets[0]], xf['GWPR_Predicted'])))
        print(r2_string + str(me.R2(xf[targets[1]], xf['Food_Predicted'])))
        print(r2_string + str(me.R2(xf[targets[2]], xf['Bird_Predicted'])))
    
    ### Save model
    if overwrite != False:
        torch.save(net, 'model.pt')

if __name__ == "__main__":
    main(overwrite=True)