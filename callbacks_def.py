import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import datams100
import pages.main_page

def register_callbacks(app):
    # Actualizar las imágenes en el panel de control
    @app.callback(
    [Output("grafica-1", "src"),
     Output("grafica-2", "src"),
     Output("grafica-3", "src"),
     Output("grafica-4", "src"),
     Output("grafica-5", "src"),
     Output("grafica-6", "src"),
     Output("grafica-7", "src"),
     Output("grafica-8", "figure"),
     Output("grafica-9", "figure")],
    [Input("update-button", "n_clicks")])

    def update_graphs(n_clicks):
        ruta = "aztechsat-decoder-python-1.0.0/beacon_test.txt"  # Ruta del archivo
        frame_data = datams100.read_hex_file(ruta)  # Leer y extraer el frame

        if frame_data != "NO DATA":
            decoder = datams100.Decoder(frame_data)  # Pasar los datos al decoder
    
            
            (fig1, fig2, fig3, 
             fig4, fig5, fig6,
             fig7)  = decoder.decode_to_graph()
            
            (fig8, fig9) = decoder.interactive_graph()

            # Convertir figuras a base64 y devolver
            return (datams100.fig_to_uri(fig1), datams100.fig_to_uri(fig2), datams100.fig_to_uri(fig3), 
                    datams100.fig_to_uri(fig4), datams100.fig_to_uri(fig5), datams100.fig_to_uri(fig6),
                    datams100.fig_to_uri(fig7), fig8, fig9)
        else:
            return tuple([""] * 9)
        
    @app.callback(
    Output("url", "pathname"),
    [Input("go-to-graphic", "n_clicks")])

    def graphic_redirect(n_clicks):
        if n_clicks is None:
            return dash.no_update # Si no ha hecho clic, no cambia nada
        return '/graphic_layout'