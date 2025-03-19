"""
Created on Wed Mar 19 13:47:59 2025

@author: robertrouse
"""

import plotly.graph_objects as pg


def single_dumbell(label, base, update, c_scale, limits):
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
            mode="markers",
            marker=dict(color="silver",
                        size=16,)),
            pg.Scatter(
                x=[label],
                y=[update],
                name='New scenario',
                mode="markers",
                marker=dict(size=32,
                            color=[update],
                            colorscale=[[c_scale[0], '#8B424B'],
                                        [c_scale[1], '#CCA857'],
                                        [c_scale[2], '#7DB567']],
                            cmin=limits[0],
                            cmax=limits[1]),)
            ]
        )
    return fig
