"""
Prediction Module
=================
This file makes predictions on new patient data.

What it does:
1. Loads the trained model
2. Takes patient information
3. Predicts if they have diabetes
4. Provides confidence score
"""

import pandas as pd
import numpy as np
import joblib
import os
from textblob import TextBlob


class DiabetesPredictor:
    """
    A class to make diabetes predictions
    """
    
    def __init__(self, model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
        """
        Load the trained model and scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("✅ Model and scaler loaded successfully!")
        
    def preprocess_input(self, patient_data):
        """
        Prepare patient data for prediction
        
        Args:
            patient_data: Dictionary with patient information
        """
        # Create DataFrame
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
        df_scaled = self.scaler.transform(df)
        
        return df_scaled
    
    def predict(self, patient_data):
        """
        Make prediction for a patient
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            prediction: 0 (No Diabetes) or 1 (Has Diabetes)
            probability: Confidence score (0-100%)
        """
        # Preprocess
        X = self.preprocess_input(patient_data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return prediction, probability
    
    def get_risk_level(self, probability):
        """
        Determine risk level based on probability
        """
        prob_diabetes = probability[1] * 100
        
        if prob_diabetes < 30:
            return "LOW RISK", "🟢"
        elif prob_diabetes < 60:
            return "MODERATE RISK", "🟡"
        else:
            return "HIGH RISK", "🔴"
    
    def get_recommendations(self, patient_data, prediction):
        """
        Provide health recommendations based on prediction
        """
        recommendations = []
        
        if prediction == 1:
            recommendations.append("⚠️ High risk of diabetes detected. Please consult a healthcare professional.")
        
        if patient_data['BMI'] > 30:
            recommendations.append("🏃 Your BMI is high. Consider regular exercise and a balanced diet.")
        
        if patient_data['Glucose'] > 140:
            recommendations.append("🍎 High glucose levels detected. Monitor your sugar intake.")
        
        if patient_data['BloodPressure'] > 90:
            recommendations.append("💊 Blood pressure is elevated. Consider stress management techniques.")
        
        if patient_data['Age'] > 50:
            recommendations.append("🩺 Regular health checkups are recommended for your age group.")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Your health indicators look good! Maintain a healthy lifestyle.")
        
        return recommendations
    
    def analyze_text_input(self, text):
        """
        Analyze text input using NLP (Natural Language Processing)
        
        This extracts health-related information from text
        """
        blob = TextBlob(text.lower())
        
        analysis = {
            'sentiment': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            'keywords': [],
            'concerns': []
        }
        
        # Health-related keywords
        health_keywords = {
            'symptoms': ['tired', 'thirsty', 'frequent urination', 'blurred vision', 
                        'hungry', 'weight loss', 'slow healing'],
            'lifestyle': ['exercise', 'diet', 'smoking', 'alcohol', 'stress'],
            'family_history': ['family', 'parent', 'mother', 'father', 'sibling', 'genetic']
        }
        
        # Find keywords in text
        for category, keywords in health_keywords.items():
            found = [word for word in keywords if word in text.lower()]
            if found:
                analysis['keywords'].append({category: found})
        
        # Detect concerns
        concern_words = ['worried', 'concerned', 'scared', 'afraid', 'problem', 'issue']
        if any(word in text.lower() for word in concern_words):
            analysis['concerns'].append("Patient expresses health concerns")
        
        return analysis
    
    def generate_report(self, patient_data, prediction, probability, text_input=None):
        """
        Generate a comprehensive health report
        """
        report = []
        report.append("="*60)
        report.append("🏥 DIABETES PREDICTION REPORT")
        report.append("="*60)
        
        # Patient Info
        report.append("\n📋 PATIENT INFORMATION:")
        report.append(f"   Age: {patient_data['Age']} years")
        report.append(f"   BMI: {patient_data['BMI']:.1f}")
        report.append(f"   Glucose: {patient_data['Glucose']} mg/dL")
        report.append(f"   Blood Pressure: {patient_data['BloodPressure']} mm Hg")
        
        # Prediction
        result = "🔴 DIABETES DETECTED" if prediction == 1 else "🟢 NO DIABETES"
        report.append(f"\n🎯 PREDICTION: {result}")
        
        # Confidence
        prob_diabetes = probability[1] * 100
        report.append(f"\n📊 CONFIDENCE SCORES:")
        report.append(f"   Diabetes: {prob_diabetes:.1f}%")
        report.append(f"   No Diabetes: {(100-prob_diabetes):.1f}%")
        
        # Risk Level
        risk, emoji = self.get_risk_level(probability)
        report.append(f"\n{emoji} RISK LEVEL: {risk}")
        
        # Recommendations
        recommendations = self.get_recommendations(patient_data, prediction)
        report.append(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            report.append(f"   {rec}")
        
        # NLP Analysis (if text provided)
        if text_input:
            nlp_analysis = self.analyze_text_input(text_input)
            report.append(f"\n🧠 TEXT ANALYSIS:")
            
            if nlp_analysis['keywords']:
                report.append("   Keywords detected:")
                for item in nlp_analysis['keywords']:
                    for category, words in item.items():
                        report.append(f"      {category}: {', '.join(words)}")
            
            if nlp_analysis['concerns']:
                report.append("   Concerns detected:")
                for concern in nlp_analysis['concerns']:
                    report.append(f"      - {concern}")
            
            sentiment = nlp_analysis['sentiment']
            if sentiment > 0.1:
                report.append("   Sentiment: Positive 😊")
            elif sentiment < -0.1:
                report.append("   Sentiment: Negative 😟")
            else:
                report.append("   Sentiment: Neutral 😐")
        
        report.append("\n" + "="*60)
        report.append("⚕️ Note: This is an AI prediction. Always consult a doctor!")
        report.append("="*60)
        
        return "\n".join(report)


def predict_single_patient():
    """
    Interactive prediction for a single patient
    """
    print("\n🏥 DIABETES PREDICTION SYSTEM")
    print("="*60)
    
    # Load predictor
    predictor = DiabetesPredictor()
    
    # Get patient data
    print("\n📝 Please enter patient information:")
    print("   (Press Enter to use default values for testing)")
    
    try:
        pregnancies = int(input("Pregnancies (0-17): ") or 2)
        glucose = float(input("Glucose (0-200 mg/dL): ") or 120)
        bp = float(input("Blood Pressure (0-122 mm Hg): ") or 70)
        skin = float(input("Skin Thickness (0-99 mm): ") or 20)
        insulin = float(input("Insulin (0-846 μU/mL): ") or 80)
        bmi = float(input("BMI (0-67.1): ") or 32)
        dpf = float(input("Diabetes Pedigree Function (0.0-2.5): ") or 0.5)
        age = int(input("Age (21-81 years): ") or 33)
        
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
        
        # Optional: Text input
        print("\n📝 Any additional health concerns? (Press Enter to skip)")
        text_input = input("Describe symptoms or concerns: ")
        
        # Make prediction
        print("\n🔮 Analyzing data...")
        prediction, probability = predictor.predict(patient_data)
        
        # Generate report
        report = predictor.generate_report(
            patient_data, prediction, probability, 
            text_input if text_input else None
        )
        
        print(report)
        
        # Save report
        save = input("\n💾 Save report to file? (y/n): ")
        if save.lower() == 'y':
            os.makedirs('reports', exist_ok=True)
            filename = f"reports/patient_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please enter valid numeric values.")


def batch_predict():
    """
    Make predictions for multiple patients from CSV
    """
    print("\n🏥 BATCH PREDICTION MODE")
    print("="*60)
    
    # Load predictor
    predictor = DiabetesPredictor()
    
    # Load patient data
    csv_path = input("\nEnter CSV file path (or press Enter for test data): ")
    if not csv_path:
        csv_path = 'data/diabetes.csv'
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} patients")
    
    # Make predictions
    predictions = []
    probabilities = []
    
    print("\n🔮 Making predictions...")
    for idx, row in df.iterrows():
        patient_data = row.drop('Outcome').to_dict() if 'Outcome' in row else row.to_dict()
        pred, prob = predictor.predict(patient_data)
        predictions.append(pred)
        probabilities.append(prob[1] * 100)
    
    # Add results to dataframe
    df['Predicted'] = predictions
    df['Probability'] = probabilities
    df['Risk_Level'] = df['Probability'].apply(
        lambda x: 'Low' if x < 30 else ('Moderate' if x < 60 else 'High')
    )
    
    # Show summary
    print("\n📊 PREDICTION SUMMARY:")
    print(f"   Total patients: {len(df)}")
    print(f"   Predicted with diabetes: {sum(predictions)}")
    print(f"   Predicted without diabetes: {len(predictions) - sum(predictions)}")
    print(f"\n   High risk: {sum(df['Risk_Level'] == 'High')}")
    print(f"   Moderate risk: {sum(df['Risk_Level'] == 'Moderate')}")
    print(f"   Low risk: {sum(df['Risk_Level'] == 'Low')}")
    
    # Save results
    output_path = 'reports/batch_predictions.csv'
    os.makedirs('reports', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path}")


# Main execution
if __name__ == "__main__":
    print("\n🏥 DIABETES PREDICTION SYSTEM")
    print("="*60)
    print("\nSelect mode:")
    print("1. Single patient prediction (Interactive)")
    print("2. Batch prediction (CSV file)")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == '1':
        predict_single_patient()
    elif choice == '2':
        batch_predict()
    else:
        print("Invalid choice!")
        