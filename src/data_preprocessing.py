"""
Data Preprocessing Module
==========================
This file cleans and prepares the diabetes dataset for machine learning.

What it does:
1. Loads the CSV file
2. Handles missing values (replaces 0s with averages)
3. Splits data into training and testing sets
4. Scales features (makes all numbers comparable)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks.
    Think of this as a data cleaning machine!
    """
    
    def __init__(self, data_path='data/diabetes.csv'):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Location of your CSV file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.df = None
        
    def load_data(self):
        """
        Load the diabetes dataset from CSV
        """
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded! Shape: {self.df.shape}")
        print(f"   (That means {self.df.shape[0]} patients, {self.df.shape[1]} features)")
        return self.df
    
    def explore_data(self):
        """
        Show basic information about the dataset
        """
        print("\n📊 Dataset Overview:")
        print("=" * 50)
        print(self.df.head())
        print("\n📈 Statistical Summary:")
        print(self.df.describe())
        print("\n🔍 Missing Values:")
        print(self.df.isnull().sum())
        print(f"\n🎯 Diabetes Cases: {self.df['Outcome'].value_counts()}")
        
    def handle_missing_values(self):
        """
        Replace 0 values with median (middle value) for certain columns
        Why? Because 0 Glucose or 0 Blood Pressure is impossible!
        """
        print("\n🔧 Handling missing values (zeros)...")
        
        # Columns that shouldn't have 0 values
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI']
        
        for column in zero_columns:
            if column in self.df.columns:
                # Replace 0 with NaN (Not a Number)
                self.df[column] = self.df[column].replace(0, np.nan)
                # Fill NaN with median value
                median_value = self.df[column].median()
                self.df[column].fillna(median_value, inplace=True)
                print(f"   ✓ Fixed {column} (used median: {median_value:.2f})")
        
        print("✅ Missing values handled!")
        
    def create_features(self):
        """
        Create new features from existing ones
        This helps the AI model learn better!
        """
        print("\n🎨 Creating new features...")
        
        # BMI Categories (Body Mass Index)
        self.df['BMI_Category'] = pd.cut(self.df['BMI'], 
                                         bins=[0, 18.5, 25, 30, 100],
                                         labels=[0, 1, 2, 3])  # 0=underweight, 1=normal, 2=overweight, 3=obese
        
        # Age Groups
        self.df['Age_Group'] = pd.cut(self.df['Age'],
                                      bins=[0, 30, 45, 60, 100],
                                      labels=[0, 1, 2, 3])  # 0=young, 1=middle, 2=senior, 3=elderly
        
        # Glucose Level Categories
        self.df['Glucose_Level'] = pd.cut(self.df['Glucose'],
                                          bins=[0, 100, 125, 200],
                                          labels=[0, 1, 2])  # 0=normal, 1=prediabetes, 2=diabetes
        
        # Convert categorical features to numbers
        self.df['BMI_Category'] = self.df['BMI_Category'].astype(float)
        self.df['Age_Group'] = self.df['Age_Group'].astype(float)
        self.df['Glucose_Level'] = self.df['Glucose_Level'].astype(float)
        
        print("✅ New features created!")
        print(f"   - BMI Category (0=underweight, 1=normal, 2=overweight, 3=obese)")
        print(f"   - Age Group (0=young, 1=middle, 2=senior, 3=elderly)")
        print(f"   - Glucose Level (0=normal, 1=prediabetes, 2=diabetes)")
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training set (80%) and testing set (20%)
        
        Why? We train the AI on 80% and test it on the remaining 20%
        to see if it really learned!
        
        Args:
            test_size: Percentage of data for testing (0.2 = 20%)
            random_state: Random seed for reproducibility
        """
        print("\n✂️ Splitting data into training and testing sets...")
        
        # Separate features (X) and target (y)
        X = self.df.drop('Outcome', axis=1)  # All columns except 'Outcome'
        y = self.df['Outcome']  # Only the 'Outcome' column
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Data split complete!")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features to have similar ranges
        
        Why? Some features are 0-1, others are 0-200. 
        Scaling makes them comparable for the AI model.
        
        Args:
            X_train: Training features
            X_test: Testing features
        """
        print("\n⚖️ Scaling features...")
        
        # Fit scaler on training data and transform both sets
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print("✅ Features scaled!")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, path='models/scaler.pkl'):
        """
        Save the scaler for later use
        
        Why? When we make predictions on new data, we need to scale it
        the same way we scaled the training data.
        """
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"\n💾 Scaler saved to {path}")
        
    def preprocess_pipeline(self):
        """
        Complete preprocessing pipeline
        Runs all steps in order!
        """
        print("\n" + "="*60)
        print("🚀 STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Create new features
        self.create_features()
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 7: Save scaler
        self.save_scaler()
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# Main execution
if __name__ == "__main__":
    """
    This runs when you execute this file directly
    """
    print("\n🏥 DIABETES PREDICTION - DATA PREPROCESSING")
    print("="*60)
    
    # Create preprocessor instance
    preprocessor = DataPreprocessor()
    
    # Run the complete pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Save processed data
    print("\n💾 Saving processed data...")
    os.makedirs('data', exist_ok=True)
    
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("✅ Processed data saved!")
    print("\n🎉 You can now proceed to model training!")
    