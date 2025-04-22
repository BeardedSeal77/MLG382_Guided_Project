# MLG382_Guided_Project

Report submission is under Documentation/

# Student Performance Analysis & Prediction

This project analyzes student performance data to identify at-risk students and provide actionable insights for BrightPath Academy. Using machine learning models, we predict student grade classifications based on various factors, with special emphasis on understanding the impact of extracurricular activities on academic outcomes.

## 📌 Problem Statement

BrightPath Academy faces several challenges:

- **Delayed identification of at-risk students:** No real-time system to identify struggling students for timely intervention
- **Lack of individual support strategies:** No tools to identify specific reasons why students are struggling
- **Unknown effect of extracurricular activities:** Uncertainty about whether extracurricular involvement impacts academic performance
- **No centralized system for actionable insights:** Abundant student data without proper analysis tools

## 💡 Key Hypotheses

Our analysis explores several hypotheses:

1. **Study Time & Grades:** Negative correlation between weekly study time and grade classification (more study → better grades)
2. **Absences & Grades:** Positive correlation between absences and grade classification (more absences → worse grades)
3. **Tutoring & Grades:** Relationship between tutoring status and grade classification
4. **Parental Support & Grades:** Negative correlation between parental support level and grade classification
5. **Extracurricular Activities & Grades:** Relationship between extracurricular participation and grade classification
6. **Parental Education & Grades:** Negative correlation between parental education level and grade classification

## 🔍 Data Analysis Workflow

1. **Data Understanding**: Initial exploration of the dataset structure
2. **Exploratory Data Analysis**: Univariate and bivariate analysis of key features
3. **Data Cleaning**: Treatment of missing values and outliers
4. **Feature Engineering**: Creating more informative features for model building
5. **Model Building**: Implementation of multiple ML models
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Deep Learning
6. **Model Deployment**: Creation of a Dash web application for predictions

## 🤖 Machine Learning Models

We developed and compared several models:
- Logistic Regression
- Random Forest Classifier 
- XGBoost Classifier
- Deep Learning (Neural Network)

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🚀 Web Application

The project includes a Dash application for real-time student performance prediction and visualization of key insights.

## Project Structure

```
student-performance/
│
├── Data/                        # all datasets from raw to scaled for the models
│   ├── Student_performance_data.csv        #raw data
│   ├── Student_performance_scaled.csv      #Treated data with scaling
│   └── Student_performance_non_scaled.csv  #Treated data without scaling
│
├── webapp/
│   ├── app.py                # Dash app entry point
│   ├── callbacks.py          # Dash callbacks
│   ├── layout.py             # Page layout with plots, controls
│   └── assets/               # CSS, images, logo
│
├── Artifacts/
│   ├── models/
│   ├── plots/
│   └── predictions/
│
├── Notebooks/
│   ├── GroupT.ipynb             #Main notebook, hold all the data ingest, understanding, EDA, etc...
│   └── Models/                    #Holds notebooks that run the models/
│       ├── Deep_Learning.ipynb
│       ├── Logistic_Regression.ipynb
│       ├── Random_Forest.ipynb
│       └── XGBoost.ipynb
│
├── requirements.txt          # Python dependencies (Render will use this)
└── README.md
```

## Installation & Usage

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`


## Contributors

- Waldo Blom (578068)
- Erin David Cullen (600531)
- Brandon Alfonso Lemmer (578062)
- Tristan James Ball (601541)

