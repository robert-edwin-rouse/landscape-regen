"""
Created on Tue Apr 21 16:38:52 2024

@author: robertrouse
"""

import pandas as pd
import numpy as np
import dash
import torch
import surrogate as sr
import visualisation as vi
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

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [html.Img(className='banner', src='assets/Banner_cropped.png'),
     dbc.Row(
         [dbc.Col(html.Div([
            html.Label('Grassland Increase', className='slider_label'),
            dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                          tooltip={'placement': 'bottom', 'always_visible': True},
                          updatemode='drag', id='grassland'),
            
            html.Label('Organic Increase', className='slider_label'),
            dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                          tooltip={'placement': 'bottom', 'always_visible': True},
                          updatemode='drag', id='organic'),
            
            html.Label('Peatland (Lower) Increase', className='slider_label'),
            dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                          tooltip={'placement': 'bottom', 'always_visible': True},
                          updatemode='drag', id='peatland_lo'),
            
            html.Label('Peatland (Upper) Increase', className='slider_label'),
            dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                          tooltip={'placement': 'bottom', 'always_visible': True},
                          updatemode='drag', id='peatland_up'),
            
            html.Label('Silvoarable Increase', className='slider_label'),
            dcc.Slider(min=0, max=1, step=0.0001, marks=slider_scale, value=0,
                          tooltip={'placement': 'bottom', 'always_visible': True},
                          updatemode='drag', id='silvoa'),
            
            html.Label('Silvopastoral Increase', className='slider_label'),
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
                          updatemode='drag', id='woodpa'),]),
            width={'size':6}),
         
         
         dbc.Col([html.Div(dcc.Graph(id='fig1',
                                     style={'height':'80vh'}))],
                 width={'size':2}),
         dbc.Col([html.Div(dcc.Graph(id='fig2',
                                     style={'height':'80vh'}))],
                 width={'size':2}),
         dbc.Col([html.Div(dcc.Graph(id='fig3',
                                     style={'height':'80vh'}))],
                 width={'size':2})
         ])
     

        
        
        

        # html.H3(id='output_values', style={'marginTop': 20}),
        
        
    ], style={'margin-left':'80px', 'margin-top':'0px', 'margin-right':'80px'})


# @app.callback(Output('my-graph', 'figure'),
@app.callback(
                # Output('output_values', 'children'),
              Output('fig1','figure'),
              Output('fig2','figure'),
              Output('fig3','figure'),
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
    base = np.zeros((8,))
    base = torch.from_numpy(base)
    base = net(base.float()).data.numpy()
    z = np.array([grassland, organic, peatland_lo, peatland_up,
                      silvoa, silvop, woodland, woodpa])
    z = torch.from_numpy(z)
    z = net(z.float()).data.numpy()
    fig1 = vi.single_dumbell('Net CO2e emissions % change',
                             base[0], z[0], [0, 0.5, 1.0], [-1, 1])
    fig1.update_xaxes(linecolor='black', mirror=True)
    fig1.update_yaxes(range=[-1.25, 1.25], linecolor='black', mirror=True,)
    fig1.update_layout(plot_bgcolor='white')
    fig2 = vi.single_dumbell('Farmland productivity % change',
                             base[1], z[1], [0, 0.5, 1.0], [-1, 1])
    fig2.update_xaxes(linecolor='black', mirror=True)
    fig2.update_yaxes(range=[-1.25, 1.25], linecolor='black', mirror=True)
    fig2.update_layout(plot_bgcolor='white')

    fig3 = vi.single_dumbell('Geometric bird species population change',
                             base[2], z[2], [0, 0.5, 1.0], [0.9, 1.2])
    fig3.update_xaxes(linecolor='black', mirror=True)
    fig3.update_yaxes(range=[0.9, 1.2], linecolor='black', mirror=True,)
    fig3.update_layout(plot_bgcolor='white')

    return [fig1, fig2, fig3]
    


# def display_value(grassland, organic, peatland_lo, peatland_up,
#                   silvoa, silvop, woodland, woodpa):
#     z = np.array([grassland, organic, peatland_lo, peatland_up,
#                       silvoa, silvop, woodland, woodpa])
#     z = torch.from_numpy(z)
#     z = net(z.float()).data.numpy()
#     nl = html.Br() # dash html component
#     lst1 = 'Relative change in net CO2e emissions : {}'.format(z[0])
#     lst2 = 'Relative change in total calorific production \
#             from UK farmland: {}'.format(z[1])
#     lst3 = 'Relative geometric change across 120 terrestrial \
#             bird species populations : {}'.format(z[2])
#     lst = [nl, lst1, nl, nl, lst2, nl, nl, lst3]
    
#     fig = make_subplots(rows=1, cols=3, shared_yaxes=False)
#     fig.add_trace(px.Bar(x=['Change in net CO2e emissions'], y=[z[0]],
#                          marker=dict(
#                              color=[z[0]],
#                              colorscale=[[0, '#7DB567'],   
#                                          [0.5, '#CCA857'], 
#                                          [1.0, '#8B424B']],
#                              cmin=-1.0,
#                              cmax=1.0)
#                          ), row=1, col=1)
#     fig.add_trace(px.Bar(x=['Change in total farmland calorific production'],
#                          y=[z[1]], marker=dict(
#                              color=[z[1]],
#                              colorscale=[[0, '#8B424B'],   
#                                          [0.5, '#CCA857'], 
#                                          [1.0, '#7DB567']],
#                              cmin=0.0,
#                              cmax=1.0),
#                          ), row=1, col=2)
    
    
    
#     fig.add_trace(px.Bar(x=['Geometric bird species population change'],
#                          y=[z[2]], marker=dict(
#                              color=[z[2]],
#                              colorscale=[[0.0, '#8B424B'],   
#                                          [0.5, '#CCA857'], 
#                                          [1.0, '#7DB567']],
#                              cmin=0.95,
#                              cmax=1.1),
#                          ), row=1, col=3)
#     fig.update_layout(yaxis1 = dict(range=[-4, 1.5]))
#     fig.update_layout(yaxis2 = dict(range=[-3, 1.5]))
#     fig.update_layout(yaxis3 = dict(range=[0.9, 1.2]))

#     # return lst, fig
#     return fig

if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port='8051')