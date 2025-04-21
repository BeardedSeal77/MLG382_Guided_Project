import dash
import os


from layout import create_layout
from callbacks import *

app = dash.Dash(__name__, assets_folder="assets", suppress_callback_exceptions=True)
server = app.server
app.title = "DL Model Dashboard"
app.layout = create_layout(app)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug = True, host = '0.0.0.0', port = port)
