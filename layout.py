from dash import dcc, html
import pandas as pd
import os

def create_layout(app):
    # Load training stats
    mean_path = os.path.join( "Artifacts", "predictions", "DL_train_mean.csv")
    std_path = os.path.join( "Artifacts", "predictions", "DL_train_std.csv")
    train_mean = pd.read_csv(mean_path)
    train_std = pd.read_csv(std_path)

    return html.Div([
        html.H1("Deep Learning Model Results", style={"textAlign": "center"}),

        html.Div([
            html.H3("Training Data Stats (Mean)"),
            dcc.Markdown(train_mean.to_markdown(index=False))
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.H3("Training Data Stats (Std Dev)"),
            dcc.Markdown(train_std.to_markdown(index=False))
        ], style={"marginBottom": "40px"}),

        html.Div([
            html.H3("Confusion Matrix"),
            html.Img(src=app.get_asset_url("DL_confusion_matrix.png"), style={"width": "80%", "maxWidth": "600px"})
        ], style={"textAlign": "center", "marginBottom": "40px"}),

        html.Div([
            html.H3("Accuracy per Epoch"),
            html.Img(src=app.get_asset_url("DL_accuracy_per_epoch.png"), style={"width": "80%", "maxWidth": "600px"})
        ], style={"textAlign": "center", "marginBottom": "40px"}),
    ])
