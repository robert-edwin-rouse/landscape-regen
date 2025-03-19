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

# Constraints for the model are defined here
from constraints import constraints

### Set plotting style parameters
ma.textstyle()


### Set global model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Network Loading
net = torch.load('model.pt', weights_only=False)
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
                             base[0], z[0], [-1, 1],
                             ['#7DB567','#CCA857','#8B424B'])
    fig1.update_xaxes(linecolor='black', mirror=True,
                      title_font=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'),
                      tickfont=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'))
    fig1.update_yaxes(range=[-1.25, 1.25], linecolor='black', mirror=True,
                      title_font=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'),
                      tickfont=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'))
    fig1.update_layout(plot_bgcolor='white')

    fig2 = vi.single_dumbell('Farmland productivity % change',
                             base[1], z[1], [-1, 1],
                             ['#8B424B','#CCA857','#7DB567'])
    fig2.update_xaxes(linecolor='black', mirror=True,
                      title_font=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'),
                      tickfont=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'))
    fig2.update_yaxes(range=[-1.25, 1.25], linecolor='black', mirror=True,
                      title_font=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'),
                      tickfont=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'))
    fig2.update_layout(plot_bgcolor='white')

    fig3 = vi.single_dumbell('Geometric bird species population change',
                             base[2], z[2], [0.9, 1.2],
                             ['#8B424B','#CCA857','#7DB567'])
    fig3.update_xaxes(linecolor='black', mirror=True,
                      title_font=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'),
                      tickfont=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'))
    fig3.update_yaxes(range=[0.9, 1.2], linecolor='black', mirror=True,
                      title_font=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'),
                      tickfont=dict(size=18, family='assets/fonts/GlacialIndifference-Bold.otf'))
    fig3.update_layout(plot_bgcolor='white')

    return [fig1, fig2, fig3]

# Enforce invariants on the sliders
@app.callback(
   # All the sliders
    [Output('grassland', 'value'), Output('organic', 'value'), Output('peatland_lo', 'value')
   , Output('silvoa', 'value'), Output('silvop', 'value'), Output('woodland', 'value')
   , Output('woodpa', 'value')],
    # All the sliders
    [Input('grassland', 'value'), Input('organic', 'value'), Input('peatland_lo', 'value')
    , Input('silvoa', 'value'), Input('silvop', 'value'),  Input('woodland', 'value')
    , Input('woodpa', 'value')])
def clamp_sum(g_val, o_val, p_lo, s_a, s_p, w_l, w_p):
    # Put the values into a model dictionary
    model = {"G": g_val, "O": o_val, "P_lo" : p_lo, "S_A": s_a
             , "S_P": s_p, "WL": w_l, "WP": w_p}
    # Apply balancing of the constraints when constraints are not satisfied
    for constraint in constraints:
        if not constraint.isSatisfied(model):
            model = constraint.balance(model)

    # Return the balanced values back to the UI
    return (model["G"], model["O"], model["P_lo"], model["S_A"], model["S_P"], model["WL"], model["WP"])

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
    # for backwards compatibility, use the `run_server` method if its defined otherwise
    # use `run`
    if "run_server" in app.__dir__():
      app.run_server(debug=True,  host='127.0.0.1', port='8051')
    else:
      app.run(debug=True, host='127.0.0.1', port='8051')