import dash
from dash import dcc, html

def create_layout(app):
    return html.Div([
        html.H1("📊 BrightPath Student Performance Dashboard"),
            dcc.Tabs([
                dcc.Tab(label="🧠 Predict Student Performance", children=[
                    html.Div([
                        # Model selection
                        html.Div([
                            html.H4("Select Prediction Model", style={"marginTop": "20px"}),
                            dcc.Dropdown(
                                id="model-selector",
                                options=model_options,
                                value="DL",  # Default to Deep Learning model
                                clearable=False,
                                style={"marginBottom": "20px"}
                            )
                        ], className="model-selector-container"),
                    
                        # Feature input form
                        html.Div([
                            html.H4("Enter Student Details"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Label(feature),
                                        create_input_component(feature, details)
                                    ], style={"margin": "10px", "width": "calc(25% - 20px)", "display": "inline-block"})
                                    for feature, details in feature_details.items()
                                ], style={"display": "flex", "flexWrap": "wrap"})
                            ], className="input-form-container"),
                        
                            # Submit button
                            html.Div([
                                html.Button(
                                    "Predict Grade", 
                                    id="submit", 
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#3498DB", 
                                        "color": "white",
                                        "padding": "10px 20px",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "fontSize": "16px",
                                        "cursor": "pointer",
                                        "marginTop": "20px",
                                        "marginBottom": "20px"
                                    }
                                )
                            ], style={"textAlign": "center"}),
                        
                            # Results display
                            html.Div([
                                html.Div(id="prediction-output", className="prediction-result"),
                                html.Div(id="confidence-output", className="confidence-result")
                            ], className="results-container")
                        ], className="form-container")

                    ], className="prediction-tab-content")
                ]),
            ])
        ])

def create_input_component(feature, details):
    """Helper function to create the appropriate input component based on feature details"""
    if details["type"] == "dropdown":
        return dcc.Dropdown(
            id=feature.lower(),
            options=details["options"],
            value=details["options"][0]["value"],
            clearable=False,
            style={"width": "100%", "marginBottom": "15px"}
        )
    else:  # Convert slider to dropdown
        if feature == "Age":
            options = [{"label": str(i), "value": i} for i in range(15, 19)]
            default = 16
        elif feature == "StudyTimeWeekly":
            options = [{"label": str(i), "value": i} for i in range(0, 21)]
            default = 10
        elif feature == "Absences":
            options = [{"label": str(i), "value": i} for i in range(0, 31)]
            default = 5
        else:
            options = [{"label": "0", "value": 0}, {"label": "1", "value": 1}]
            default = 0
            
        return dcc.Dropdown(
            id=feature.lower(),
            options=options,
            value=default,
            clearable=False,
            style={"width": "100%", "marginBottom": "15px"}
        )

# -------------------------------------------------------------------------------------
# Features for the drop downs
# Define the feature names and their options for dropdowns
feature_details = {
    "Age": {"type": "slider", "min": 15, "max": 18, "step": 1, "default": 16},

    "Gender": {"type": "dropdown", "options": [
        {"label": "Male", "value": 0},
        {"label": "Female", "value": 1}
    ]},
    

    "ParentalEducation": {"type": "dropdown", "options": [
        {"label": "None", "value": 0},
        {"label": "High School", "value": 1},
        {"label": "Some College", "value": 2},
        {"label": "Bachelor's", "value": 3},
        {"label": "Higher Study", "value": 4}
    ]},

    "StudyTimeWeekly": {"type": "slider", "min": 0, "max": 20, "step": 1, "default": 10},
    
    "Absences": {"type": "slider", "min": 0, "max": 30, "step": 1, "default": 5},
    
    "Tutoring": {"type": "dropdown", "options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1}
    ]},

    "ParentalSupport": {"type": "dropdown", "options": [
        {"label": "None", "value": 0},
        {"label": "Low", "value": 1},
        {"label": "Moderate", "value": 2},
        {"label": "High", "value": 3},
        {"label": "Very High", "value": 4}
    ]},

    "Extracurricular": {"type": "dropdown", "options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1}
    ]},

    "Sports": {"type": "dropdown", "options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1}
    ]},

    "Music": {"type": "dropdown", "options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1}
    ]},

    "Volunteering": {"type": "dropdown", "options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1}
    ]}
}

# Model options
model_options = [
    {"label": "Deep Learning Model", "value": "DL"},
    {"label": "Logistic Regression Model", "value": "LR"},
    {"label": "Random Forest Model", "value": "RF"},
    {"label": "XGBoost Model", "value": "XG"}
]