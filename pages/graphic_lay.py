import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from datams100 import Decoder

dash.register_page(__name__, path = "/graphic_layout")
decoder = Decoder()
obc_magneto, obc_gyro = decoder.interactive_graph()

layout = html.Div(children=[
    html.H1("Pagina"),
    html.H2("Gráficas de AX100"),
    html.Button("Actualizar Gráficas", id="update-button"),
    html.Div([
        html.Img(id="grafica-1"),
        html.Img(id="grafica-2"),
        html.Img(id="grafica-3")
    ], className="graph-container"),
    html.Div([
        html.H2("Gráficas de Pylons"),
        html.Img(id="grafica-4")
    ]),
    
    html.H2("Gráficas de OBC"),
    html.Div([
        html.Img(id="grafica-5"),
        html.Img(id="grafica-6"),
        html.Img(id="grafica-7"),
        dcc.Graph(id="grafica-8", figure= obc_magneto),
        dcc.Graph(id="grafica-9", figure= obc_gyro),
    ],className="graph-container")
])