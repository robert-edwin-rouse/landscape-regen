"""
Created on Wed Mar 19 13:47:59 2025

@author: robertrouse
"""

import plotly.graph_objects as pg


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
            marker=dict(color='silver',
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
                                width=16),
                                color=[update],
                                colorscale=[[scaling[0], colorscale[0]],
                                            [scaling[1], colorscale[1]],
                                            [scaling[2], colorscale[2]]],
                                cmin=limits[0],
                                cmax=limits[1]),)
            ]
        )
    return fig
