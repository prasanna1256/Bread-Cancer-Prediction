import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

def clean_data():
    data=pd.read_csv("data.csv")
    data.drop("Unnamed: 32", axis=1, inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    data.drop(["id"], axis=1, inplace=True)
    return data

def side_bar(data):
    st.sidebar.markdown("<h1>Input Features</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<style>h1{color:skyblue;}</style>", unsafe_allow_html=True)
    st.sidebar.write("Adjust the sliders to input feature values:")
    
    slider_values = {
    # Mean features
    "mean_radius": "radius_mean",
    "mean_texture": "texture_mean",
    "mean_perimeter": "perimeter_mean",
    "mean_area": "area_mean",
    "mean_smoothness": "smoothness_mean",
    "mean_compactness": "compactness_mean",
    "mean_concavity": "concavity_mean",
    "mean_concave_points": "concave points_mean",
    "mean_symmetry": "symmetry_mean",
    "mean_fractal_dimension": "fractal_dimension_mean",
    
    # Standard error features
    "radius_se": "radius_se",
    "texture_se": "texture_se",
    "perimeter_se": "perimeter_se",
    "area_se": "area_se",
    "smoothness_se": "smoothness_se",
    "compactness_se": "compactness_se",
    "concavity_se": "concavity_se",
    "concave_points_se": "concave points_se",
    "symmetry_se": "symmetry_se",
    "fractal_dimension_se": "fractal_dimension_se",
    
    # Worst features
    "worst_radius": "radius_worst",
    "worst_texture": "texture_worst",
    "worst_perimeter": "perimeter_worst",
    "worst_area": "area_worst",
    "worst_smoothness": "smoothness_worst",
    "worst_compactness": "compactness_worst",
    "worst_concavity": "concavity_worst",
    "worst_concave_points": "concave points_worst",
    "worst_symmetry": "symmetry_worst",
    "worst_fractal_dimension": "fractal_dimension_worst"
    }
    
    input_dict={}
    for label,key in slider_values.items():
        input_dict[key]=st.sidebar.slider(key,min_value=float(0),max_value=float(data[key].max()),value=float(data[key].mean()))
    return input_dict

def get_scaler_data(input_dict):
    data=clean_data()
    X=data.drop("diagnosis",axis=1)
    
    scaled_dict={}
    for key,value in input_dict.items():
        max_val=X[key].max()
        min_val=X[key].min()
        scaled_value=(value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict
        
def plt_radar(input_dict):
    
    input_dict=get_scaler_data(input_dict)
    categories = [
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", 
        "Mean Smoothness", "Mean Compactness", "Mean Concavity", 
        "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension"
    ]
    
    # Create radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            input_dict["radius_mean"], input_dict["texture_mean"],
            input_dict["perimeter_mean"], input_dict["area_mean"],
            input_dict["smoothness_mean"], input_dict["compactness_mean"],
            input_dict["concavity_mean"], input_dict["concave points_mean"],
            input_dict["symmetry_mean"], input_dict["fractal_dimension_mean"]
        ],
        theta=categories,
        fill='toself',
        name='Mean Features'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_dict["radius_se"], input_dict["texture_se"], input_dict["perimeter_se"],
            input_dict["area_se"], input_dict["smoothness_se"], input_dict["compactness_se"],
            input_dict["concavity_se"], input_dict["concave points_se"],
            input_dict["symmetry_se"], input_dict["fractal_dimension_se"]
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_dict["radius_worst"], input_dict["texture_worst"], input_dict["perimeter_worst"],
            input_dict["area_worst"], input_dict["smoothness_worst"], input_dict["compactness_worst"],
            input_dict["concavity_worst"], input_dict["concave points_worst"],
            input_dict["symmetry_worst"], input_dict["fractal_dimension_worst"]
        ],
        theta=categories,
        fill='toself',
        name='Worst Features'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Adjust range based on your data
            )
        ),
        showlegend=True,
        title="Breast Cancer Feature Comparison (Radar Chart)"
    )

    return fig



def add_predictions(input_dict):
    model=pickle.load(open("model.pkl", "rb"))
    scaler=pickle.load(open("scaler.pkl", "rb"))
    input_array=np.array(list(input_dict.values())).reshape(1,-1)
    
    input_array_scaled=scaler.transform(input_array)
    prediction=model.predict(input_array_scaled)
    st.markdown("<h3>Cell Cluster Prediction</h3>", unsafe_allow_html=True)
    st.write("The model predicts whether the cell cluster is **malignant** or **benign**.")
    if prediction[0]==0:
        st.success("benign!")
    else:
        st.error("maligant!")
    st.write("The probability of being healthy",model.predict_proba(input_array_scaled)[0][0])
    st.write("The probability of being diseased",model.predict_proba(input_array_scaled)[0][1])
def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=":lady-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    data = clean_data()
    input_dict = side_bar(data)
    
    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("This app predicts whether a human has breast cancer or not based on the input features.")
    
    col1, col2 = st.columns([4, 1])  # Removed `border=True`
    with col1:
        res_fig = plt_radar(input_dict)
        st.plotly_chart(res_fig, use_container_width=True)
    with col2:
        add_predictions(input_dict)

main()