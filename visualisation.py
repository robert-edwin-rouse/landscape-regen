"""
Created on Wed Mar 19 13:47:59 2025

@author: robertrouse
"""

import plotly.graph_objects as pg
import plotly.express as px


def single_dumbell(label, base, update, limits, colorscale, scaling=[0, 0.5, 1]):
    fig = pg.Figure(
        data=[
            pg.Scatter(
                x=[label, label],
                y=[base, update],
                mode='markers+lines',
                showlegend=False,
                marker=dict(symbol='arrow',
                            color='black',
                            size=16,
                            angleref='previous',
                            standoff=8)),
            pg.Scatter(x=[label],
            y=[base],
            name='2015',
            mode='markers',
            showlegend=False,
            marker=dict(color='black',
                        size=16,)),
            pg.Scatter(
                x=[label],
                y=[update],
                name='New scenario',
                mode='markers',
                showlegend=False,
                marker=dict(size=32,
                            symbol='circle-open',
                            line=dict(
                                width=8),
                                color=[update],
                                colorscale=[[scaling[0], colorscale[0]],
                                            [scaling[1], colorscale[1]],
                                            [scaling[2], colorscale[2]]],
                                cmin=limits[0],
                                cmax=limits[1]),)
            ]
        )
    return fig

def dashboard_pareto_scatter(label, pareto_x, pareto_y, pareto_z, 
                             new_x, new_y,
                             limits, colorscale, scaling=[0, 0.5, 1]):    
    fig_a = px.scatter(x=new_x, y=new_y)
    fig_a.update_traces(marker=dict(size=20, color='black',
                                    symbol="x"))
    
    fig_b = px.scatter(x=pareto_x, y=pareto_y)
    fig_b.update_traces(marker=dict(size=2.5, color=pareto_z, 
                                  colorscale=[[scaling[0], colorscale[0]],
                                              [scaling[1], colorscale[1]],
                                              [scaling[2], colorscale[2]]],
                                  cmin=limits[0],
                                  cmax=limits[1],
                                  showscale=True,
                                  colorbar_x=-0.3), opacity=0.4
                     )
    fig = pg.Figure(data=fig_a.data + fig_b.data)
    return fig