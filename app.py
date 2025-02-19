import dash
from dash import dcc, html
from dash.dependencies import Input, Output
# Layout de la app
app = dash.Dash(__name__, use_pages=True)

from callbacks_def import register_callbacks 

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dash.page_container #Carga contenido dinámicamente
])

register_callbacks(app)

# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True)