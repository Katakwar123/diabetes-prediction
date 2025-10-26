"""
Streamlit Web Application
==========================
This creates a beautiful web interface for diabetes prediction.

What it does:
1. Creates a user-friendly web interface
2. Allows users to input patient data
3. Shows predictions and visualizations
4. Includes NLP-based text analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from textblob import TextBlob
import os


# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run data preprocessing and model training first!")
        return None, None


def preprocess_input(patient_data, scaler):
    """Prepare patient data for prediction"""
    df = pd.DataFrame([patient_data])
    
    # Create additional features
    df['BMI_Category'] = pd.cut(df['BMI'], 
                                bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3]).astype(float)
    
    df['Age_Group'] = pd.cut(df['Age'],
                            bins=[0, 30, 45, 60, 100],
                            labels=[0, 1, 2, 3]).astype(float)
    
    df['Glucose_Level'] = pd.cut(df['Glucose'],
                                 bins=[0, 100, 125, 200],
                                 labels=[0, 1, 2]).astype(float)
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled


def analyze_text(text):
    """Analyze text input using NLP"""
    blob = TextBlob(text.lower())
    
    analysis = {
        'sentiment': blob.sentiment.polarity,
        'keywords': [],
        'concerns': []
    }
    
    # Health-related keywords
    health_keywords = {
        'symptoms': ['tired', 'thirsty', 'frequent urination', 'blurred vision', 
                    'hungry', 'weight loss', 'slow healing', 'fatigue'],
        'lifestyle': ['exercise', 'diet', 'smoking', 'alcohol', 'stress', 'sleep'],
        'family_history': ['family', 'parent', 'mother', 'father', 'sibling', 'genetic', 'hereditary']
    }
    
    # Find keywords
    for category, keywords in health_keywords.items():
        found = [word for word in keywords if word in text.lower()]
        if found:
            analysis['keywords'].append({category: found})
    
    # Detect concerns
    concern_words = ['worried', 'concerned', 'scared', 'afraid', 'problem', 'issue', 'help']
    if any(word in text.lower() for word in concern_words):
        analysis['concerns'].append("Patient expresses health concerns")
    
    return analysis


def create_gauge_chart(probability):
    """Create a gauge chart for diabetes risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk (%)", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'yellow'},
                {'range': [60, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60}}))
    
    fig.update_layout(height=300)
    return fig


def create_feature_comparison(patient_data):
    """Create radar chart comparing patient data to healthy ranges"""
    categories = ['Glucose', 'BMI', 'Blood Pressure', 'Age']
    
    # Normalize values to 0-100 scale for visualization
    patient_values = [
        min(patient_data['Glucose'] / 2, 100),  # Max 200 -> 100
        min(patient_data['BMI'] * 2, 100),      # Max 50 -> 100
        min(patient_data['BloodPressure'] / 1.2, 100),  # Max 120 -> 100
        min(patient_data['Age'] / 0.8, 100)     # Max 80 -> 100
    ]
    
    healthy_values = [60, 45, 65, 50]  # Healthy reference ranges
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=categories,
        fill='toself',
        name='Patient',
        line_color='red'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=healthy_values,
        theta=categories,
        fill='toself',
        name='Healthy Range',
        line_color='green'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=400
    )
    
    return fig


def generate_wordcloud(text):
    """Generate word cloud from text"""
    if text:
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    return None


# Main app
def main():
    # Header
    st.title("🏥 Diabetes Prediction System")
    st.markdown("### AI-Powered Health Assessment with NLP Analysis")
    st.markdown("---")
    
    # Load models
    model, scaler = load_models()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Go to", 
                           ["🔮 Prediction", "📊 Data Analysis", "ℹ️ About"])
    
    if page == "🔮 Prediction":
        show_prediction_page(model, scaler)
    elif page == " Data Analysis":
        show_analysis_page()
    else:
        show_about_page()


def show_prediction_page(model, scaler):
    """Prediction page"""
    st.header("🔮 Make a Prediction")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Patient Information")
        
        # Input fields
        col_a, col_b = st.columns(2)
        
        with col_a:
            pregnancies = st.number_input("Pregnancies", 0, 17, 1, 
                                         help="Number of times pregnant")
            glucose = st.number_input("Glucose Level (mg/dL)", 0, 200, 120,
                                     help="Plasma glucose concentration")
            bp = st.number_input("Blood Pressure (mm Hg)", 0, 122, 70,
                                help="Diastolic blood pressure")
            skin = st.number_input("Skin Thickness (mm)", 0, 99, 20,
                                  help="Triceps skin fold thickness")
        
        with col_b:
            insulin = st.number_input("Insulin (μU/mL)", 0, 846, 80,
                                     help="2-Hour serum insulin")
            bmi = st.number_input("BMI", 0.0, 67.1, 32.0,
                                 help="Body mass index")
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5,
                                 help="Genetic influence score")
            age = st.number_input("Age (years)", 21, 81, 33)
        
        # Text input for NLP
        st.subheader("💬 Additional Information (Optional)")
        text_input = st.text_area(
            "Describe any symptoms, concerns, or relevant medical history:",
            placeholder="e.g., 'I feel tired all the time and very thirsty. My father has diabetes.'",
            height=100
        )
    
    with col2:
        st.subheader("📊 Health Indicators")
        
        # BMI Category
        if bmi < 18.5:
            st.info("BMI: Underweight")
        elif bmi < 25:
            st.success("BMI: Normal")
        elif bmi < 30:
            st.warning("BMI: Overweight")
        else:
            st.error("BMI: Obese")
        
        # Glucose Level
        if glucose < 100:
            st.success("Glucose: Normal")
        elif glucose < 125:
            st.warning("Glucose: Prediabetes range")
        else:
            st.error("Glucose: Diabetes range")
        
        # Blood Pressure
        if bp < 80:
            st.success("Blood Pressure: Normal")
        elif bp < 90:
            st.warning("Blood Pressure: Elevated")
        else:
            st.error("Blood Pressure: High")
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Prepare data
        patient_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        # Make prediction
        X = preprocess_input(patient_data, scaler)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Display results
        st.markdown("---")
        st.header("📋 Prediction Results")
        
        # Main result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("### 🔴 DIABETES DETECTED")
            else:
                st.success("### 🟢 NO DIABETES")
        
        with col2:
            st.metric("Diabetes Probability", f"{probability[1]*100:.1f}%")
        
        with col3:
            risk = "LOW" if probability[1] < 0.3 else ("MODERATE" if probability[1] < 0.6 else "HIGH")
            risk_color = "🟢" if probability[1] < 0.3 else ("🟡" if probability[1] < 0.6 else "🔴")
            st.metric("Risk Level", f"{risk_color} {risk}")
        
        # Gauge chart
        st.plotly_chart(create_gauge_chart(probability[1]), use_container_width=True)
        
        # Feature comparison
        st.subheader("📊 Health Profile Comparison")
        st.plotly_chart(create_feature_comparison(patient_data), use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = []
        
        if prediction == 1:
            recommendations.append("⚠️ **High risk of diabetes detected.** Please consult a healthcare professional immediately.")
        
        if bmi > 30:
            recommendations.append("🏃 **BMI is high.** Consider regular exercise (30 min/day) and a balanced diet.")
        
        if glucose > 140:
            recommendations.append("🍎 **High glucose levels detected.** Monitor sugar intake and avoid processed foods.")
        
        if bp > 90:
            recommendations.append("💊 **Blood pressure is elevated.** Practice stress management and reduce salt intake.")
        
        if age > 50:
            recommendations.append("🩺 **Regular health checkups recommended** for your age group (every 6 months).")
        
        if len(recommendations) == 0:
            recommendations.append("✅ **Your health indicators look good!** Maintain a healthy lifestyle with regular exercise and balanced diet.")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # NLP Analysis
        if text_input:
            st.markdown("---")
            st.subheader("🧠 Text Analysis (NLP)")
            
            analysis = analyze_text(text_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment
                sentiment = analysis['sentiment']
                if sentiment > 0.1:
                    st.success("😊 Sentiment: Positive")
                elif sentiment < -0.1:
                    st.error("😟 Sentiment: Negative - Patient may be worried")
                else:
                    st.info("😐 Sentiment: Neutral")
                
                # Keywords
                if analysis['keywords']:
                    st.write("**Keywords Detected:**")
                    for item in analysis['keywords']:
                        for category, words in item.items():
                            st.write(f"- **{category.title()}:** {', '.join(words)}")
            
            with col2:
                # Word cloud
                fig = generate_wordcloud(text_input)
                if fig:
                    st.pyplot(fig)
            
            # Concerns
            if analysis['concerns']:
                st.warning("⚠️ " + " | ".join(analysis['concerns']))
        
        # Disclaimer
        st.markdown("---")
        st.info("⚕️ **Disclaimer:** This is an AI prediction tool for educational purposes. Always consult qualified healthcare professionals for medical advice.")


def show_analysis_page():
    """Data analysis page"""
    st.header("📊 Dataset Analysis")
    
    # Load dataset
    try:
        df = pd.read_csv('data/diabetes.csv')
        
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Diabetes Cases", df['Outcome'].sum())
        with col3:
            st.metric("No Diabetes", len(df) - df['Outcome'].sum())
        with col4:
            st.metric("Features", len(df.columns) - 1)
        
        # Distribution plots
        st.subheader("📈 Feature Distributions")
        
        feature = st.selectbox("Select Feature", 
                              ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin'])
        
        fig = px.histogram(df, x=feature, color='Outcome',
                          nbins=30, 
                          title=f'{feature} Distribution by Outcome',
                          color_discrete_map={0: 'green', 1: 'red'},
                          labels={'Outcome': 'Diabetes'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlations")
        corr = df.corr()
        fig = px.imshow(corr, 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def show_about_page():
    """About page"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🏥 Diabetes Prediction System
    
    This is an AI-powered health assessment tool that predicts diabetes risk using machine learning.
    
    #### 🎯 Features:
    - **Machine Learning Prediction**: Uses trained ML models (Random Forest, Gradient Boosting, etc.)
    - **NLP Text Analysis**: Analyzes patient descriptions using Natural Language Processing
    - **Interactive Visualizations**: Beautiful charts and gauges
    - **Health Recommendations**: Personalized advice based on risk factors
    
    #### 📊 Dataset:
    - **Source**: Pima Indians Diabetes Database
    - **Features**: 8 medical predictors (Glucose, BMI, Age, etc.)
    - **Samples**: 768 patients
    
    #### 🤖 Model Performance:
    - **Accuracy**: ~75-80%
    - **Algorithm**: Ensemble methods (Random Forest/Gradient Boosting)
    
    #### 🛠️ Technology Stack:
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning
    - **Streamlit**: Web interface
    - **TextBlob**: NLP analysis
    - **Plotly**: Interactive visualizations
    
    #### ⚠️ Important Notice:
    This tool is for **educational purposes only**. It should **not** replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.
    
    #### 👨‍💻 For Developers:
    This project demonstrates:
    - Data preprocessing and feature engineering
    - Model training and evaluation
    - Web application development
    - NLP integration
    - Interactive data visualization
    
    ---
    # Developed by Balkrishna Katakwar
© 2025 Balkrishna Katakwar. All rights reserved.
Author: Balkrishna Katakwar
End of script.
    """)


if __name__ == "__main__":
    main()
    