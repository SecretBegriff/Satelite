import dash
from dash import dcc, html
from dash.dependencies import Input, Output


dash.register_page(__name__, path = "/")

layout = html.Div(children=[
    html.H1("Pagina Principal"),
    html.H2("¡¡MS100!!"),
    html.Button("Ir a Graficas", id="go-to-graphic"),
    dcc.Location(id='url', refresh=True)])

