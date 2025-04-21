import dash
from dash import callback, Input, Output, State, dcc, html
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import base64
import io
import os
from dash.exceptions import PreventUpdate

# Define mapping for grade classes
grade_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'F'
}

def get_grade_color(grade_class):
    """Return color based on grade class"""
    colors = {
        0: "#28a745",  # A - Green
        1: "#17a2b8",  # B - Teal
        2: "#ffc107",  # C - Yellow
        3: "#fd7e14",  # D - Orange
        4: "#dc3545"   # F - Red
    }
    return colors.get(grade_class, "#6c757d")

# Cache for loaded models to avoid reloading
model_cache = {}

def get_model(model_type):
    """Load and cache the appropriate model based on selection"""
    if model_type in model_cache:
        return model_cache[model_type]
    
    if model_type == "DL":
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="./Artifacts/models/DL_model.tflite")
        interpreter.allocate_tensors()
        model_cache[model_type] = {
            "type": "tflite",
            "model": interpreter
        }
    else:
        # Load other models (pickle files)
        model_path = f"./Artifacts/models/{model_type}_model.pkl"
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        model_cache[model_type] = {
            "type": "sklearn",
            "model": loaded_model
        }
    
    return model_cache[model_type]

def load_scaling_params():
    """Load mean and std deviation for feature scaling"""
    train_mean = pd.read_csv("./Artifacts/predictions/DL_train_mean.csv", index_col=0).squeeze("columns")
    train_std = pd.read_csv("./Artifacts/predictions/DL_train_std.csv", index_col=0).squeeze("columns")
    return train_mean, train_std



# Register callbacks directly with Dash application
@callback(
    [Output("prediction-output", "children"),
     Output("confidence-output", "children")],
    
    Input("submit", "n_clicks"),
    [State("model-selector", "value"),
     State("age", "value"),
     State("gender", "value"),
     State("parentaleducation", "value"),
     State("studytimeweekly", "value"),
     State("absences", "value"),
     State("tutoring", "value"),
     State("parentalsupport", "value"),
     State("extracurricular", "value"),
     State("sports", "value"),
     State("music", "value"),
     State("volunteering", "value")]
)
def predict_grade(n_clicks, model_type, age, gender, parental_education, 
                  study_time, absences, tutoring, parental_support, 
                  extracurricular, sports, music, volunteering):
    """Predict student's grade based on input features"""
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
        
    if None in [age, gender, parental_education, study_time, absences, 
                tutoring, parental_support, extracurricular, sports, music, volunteering]:
        return "Please fill in all input fields.", ""
    
    # Prepare input data
    feature_names = [
        "Age", "Gender", "ParentalEducation",
        "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
        "Extracurricular", "Sports", "Music", "Volunteering"
    ]
    
    input_data = [age, gender, parental_education, study_time, absences, 
                  tutoring, parental_support, extracurricular, sports, music, volunteering]
    
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Load scaling parameters
    train_mean, train_std = load_scaling_params()
    
    # Scale input data - only use the columns that exist in the input data
    scaled_mean = train_mean[train_mean.index.isin(feature_names)]
    scaled_std = train_std[train_std.index.isin(feature_names)]
    input_scaled = (input_df - scaled_mean) / scaled_std
    
    # Get model
    model_info = get_model(model_type)
    
    # Make prediction
    if model_info["type"] == "tflite":
        interpreter = model_info["model"]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_array = np.array(input_scaled, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_class]) * 100
    else:
        model = model_info["model"]
        predicted_class = model.predict(input_scaled)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            confidence = float(proba[predicted_class]) * 100
        else:
            confidence = None
    
    # Format result
    result_div = html.Div([
        html.H3("Prediction Result:", style={"marginBottom": "10px"}),
        html.Div([
            html.Span("Predicted Grade: ", style={"fontWeight": "bold"}),
            html.Span(f"{grade_mapping[predicted_class]}", 
                     style={"fontSize": "24px", "color": get_grade_color(predicted_class)})
        ], style={"fontSize": "18px"})
    ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
    
    confidence_div = ""
    if confidence is not None:
        confidence_div = html.Div([
            html.P(f"Confidence: {confidence:.2f}%", 
                  style={"fontSize": "16px", "marginTop": "10px"})
        ], style={"textAlign": "center"})
    
    return result_div, confidence_div