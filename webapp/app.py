import dash
import os

app = dash.Dash(__name__)
server = app.server


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug = True, host = '0.0.0.0', port = port)