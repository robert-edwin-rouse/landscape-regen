"""
Created on Tue Apr 21 16:38:52 2024

@author: robertrouse
"""

import pandas as pd
import numpy as np
import dash
import torch
import surrogate as sr
import matplotlib.pyplot as plt
from torch.autograd import Variable
from apollo import mechanics as ma
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as px


### Set plotting style parameters
ma.textstyle()


### Set global model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Network Loading
net = torch.load('model.pt')
net.eval()


slider_scale = {0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 0.4:'0.4', 0.5:'0.5',
                0.6:'0.6', 0.7:'0.7', 0.8:'0.8', 0.9:'0.9', 1.0:'1.0'}

app = Dash(__name__)
app.layout = html.Div(
    [   html.Img(className='banner', src='assets/Banner.png'),

        html.Label('Grassland Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='grassland'),
        
        html.Label('Organic Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='organic'),
        
        html.Label('Peatland Low Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='peatland_lo'),
        
        html.Label('Peatland Up Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='peatland_up'),
        
        html.Label('Silvoa Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='silvoa'),
        
        html.Label('Silvop Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='silvop'),
        
        html.Label('Woodland Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='woodland'),
        
        html.Label('Wood Pasture Increase', className='slider_label'),
        dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                      tooltip={'placement': 'bottom', 'always_visible': True},
                      updatemode='drag', id='woodpa'),
        
        dcc.Graph(id='output_graph'),

        html.H3(id='output_values', style={'marginTop': 20}),
        
        
    ], style={"margin": 20})


# @app.callback(Output('my-graph', 'figure'),
@app.callback(Output('output_values', 'children'),
              Output('output_graph','figure'),
              Input('grassland', 'value'),
              Input('organic', 'value'),
              Input('peatland_lo', 'value'),
              Input('peatland_up', 'value'),
              Input('silvoa', 'value'),
              Input('silvop', 'value'),
              Input('woodland', 'value'),
              Input('woodpa', 'value'),
              )

def display_value(grassland, organic, peatland_lo, peatland_up,
                  silvoa, silvop, woodland, woodpa):
    z = np.array([grassland, organic, peatland_lo, peatland_up,
                      silvoa, silvop, woodland, woodpa])
    z = torch.from_numpy(z)
    z = net(z.float()).data.numpy()
    nl = html.Br() # dash html component
    lst1 = 'Relative change in net CO2e emissions : {}'.format(z[0])
    lst2 = 'Relative change in total calorific production \
            from UK farmland: {}'.format(z[1])
    lst3 = 'Relative geometric change across 120 terrestrial \
            bird species populations : {}'.format(z[2])
    lst = [nl, lst1, nl, nl, lst2, nl, nl, lst3]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=False)
    fig.add_trace(px.Bar(x=['Change in net CO2e emissions'],
                         y=[z[0]],
                         ), row=1, col=1)
    fig.add_trace(px.Bar(x=['Change in total farmland calorific production'],
                         y=[z[1]],
                         ), row=1, col=2)
    fig.add_trace(px.Bar(x=['Geometric bird species population change'],
                         y=[z[2]],
                         ), row=1, col=3)
    fig.update_layout(yaxis1 = dict(range=[-4, 1.5]))
    fig.update_layout(yaxis2 = dict(range=[-3, 1.5]))
    fig.update_layout(yaxis3 = dict(range=[0.9, 1.2]))

    return lst, fig

if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port='8051')