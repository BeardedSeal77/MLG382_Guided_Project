import dash
from dash import callback, Input, Output, State, html
import pandas as pd
import numpy as np
import tensorflow as tf
from dash.exceptions import PreventUpdate

# Define mapping for grade classes
grade_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'F'
}

# Return color based on grade class
def get_grade_color(grade_class):

    colors = {
        0: "#28a745",  # A - Green
        1: "#17a2b8",  # B - Teal
        2: "#ffc107",  # C - Yellow
        3: "#fd7e14",  # D - Orange
        4: "#dc3545"   # F - Red
    }
    return colors.get(grade_class, "#6c757d")

# Load the model at startup
interpreter = tf.lite.Interpreter(model_path="./Artifacts/models/DL_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print expected input shape for debugging
print("Expected input shape:", input_details[0]['shape'])

# missing the scaling parameters, initialize with zeros
feature_names = [
    "Age", "Gender", "ParentalEducation", 
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering",
    "StudyAbsenceRatio", "SportsMusic", "TotalSupport"
]

# Create default scaling parameters (no scaling)
train_mean = pd.Series(0, index=feature_names)
train_std = pd.Series(1, index=feature_names)

# Try to load from files if they exist
try: # Massive debugging because scaling values not loading on render
    loaded_mean = pd.read_csv("./Artifacts/predictions/DL_train_mean.csv", index_col=0).squeeze("columns")
    loaded_std = pd.read_csv("./Artifacts/predictions/DL_train_std.csv", index_col=0).squeeze("columns")
    
    # Update our default series with any values that exist in the files
    for feature in loaded_mean.index:
        if feature in train_mean.index:
            train_mean[feature] = loaded_mean[feature]
            train_std[feature] = loaded_std[feature]
    
    print("Loaded scaling parameters for:", list(loaded_mean.index))
except Exception as e:
    print(f"Warning: Could not load scaling parameters: {e}")
    # Continue with default scaling (no scaling)

@callback(
    Output("prediction-output", "children"),
    Input("submit", "n_clicks"),
    [State("age", "value"),
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

# Predict student's grade based on input features
def predict_grade(n_clicks, age, gender, parental_education, study_time_weekly, 
                 absences, tutoring, parental_support, extracurricular, sports, music, volunteering):
    if not n_clicks:
        raise PreventUpdate

    if None in [age, gender, parental_education, study_time_weekly, absences, tutoring, 
                parental_support, extracurricular, sports, music, volunteering]:
        return html.Div("Please fill in all input fields.", 
                       style={"color": "#dc3545", "fontWeight": "bold"})

    try:
        # Calculate derived features
        study_absence_ratio = study_time_weekly / (absences + 1)
        sports_music = sports * music
        total_support = parental_support + tutoring

        # Create input data with all features - as a simple list first
        input_values = [
            age, gender, parental_education, study_time_weekly, absences, 
            tutoring, parental_support, extracurricular, sports, music, 
            volunteering, study_absence_ratio, sports_music, total_support
        ]
        
        print("Input values:", input_values)
        
        # Create a proper numpy array with shape (1, 14)
        input_array = np.array([input_values], dtype=np.float32)
        
        print("Initial array shape:", input_array.shape)
        
        # Scale the input features manually
        for i, feature in enumerate(feature_names):
            input_array[0, i] = (input_array[0, i] - train_mean[feature]) / train_std[feature]
            
        # Fill any NaN values with zeros
        input_array = np.nan_to_num(input_array)
        
        print("Scaled array shape:", input_array.shape)
        print("Scaled array:", input_array)
        
        # Set tensor and invoke model
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get prediction and confidence
        predicted_class = int(np.argmax(output_data))
        confidence = float(np.max(output_data)) * 100
        grade = grade_mapping[predicted_class]
        color = get_grade_color(predicted_class)
        
        # Build simple result UI
        result_div = html.Div([
            html.H3("Prediction Result"),
            html.P(f"Predicted Grade: {grade}", style={
                "fontSize": "24px",
                "color": color,
                "fontWeight": "bold"
            }),
            html.P(f"Confidence: {confidence:.2f}%", style={
                "fontSize": "16px", 
                "marginTop": "10px"
            }),
            html.Div([
                html.H4("Calculated Features"),
                html.P(f"Study/Absence Ratio: {study_absence_ratio:.2f}"),
                html.P(f"Sports & Music: {sports_music}"),
                html.P(f"Total Support: {total_support}")
            ], style={"marginTop": "15px", "textAlign": "left"})
        ], style={
            "textAlign": "center",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px"
        })

        return result_div
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Return the error message to display to the user
        return html.Div([
            html.H3("Error", style={"color": "#dc3545"}),
            html.P(f"Error during prediction: {str(e)}"),
            html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap"})
        ], style={
            "textAlign": "left",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px",
            "border": "1px solid #dc3545"
        })