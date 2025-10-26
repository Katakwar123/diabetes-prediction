"""
Model Training Module
=====================
This file trains multiple machine learning models to predict diabetes.

What it does:
1. Trains 5 different AI models
2. Compares their performance
3. Saves the best model
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A class to train and compare multiple ML models
    """
    
    def __init__(self):
        """
        Initialize with different models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'Support Vector Machine': SVC(random_state=42, probability=True)
        }
        
        self.trained_models = {}
        self.results = {}
        
    def load_data(self):
        """
        Load the preprocessed data
        """
        print("📂 Loading preprocessed data...")
        
        X_train = pd.read_csv('data/X_train.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_train = pd.read_csv('data/y_train.csv').values.ravel()
        y_test = pd.read_csv('data/y_test.csv').values.ravel()
        
        print(f"✅ Data loaded!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, name, model, X_train, y_train, X_test, y_test):
        """
        Train a single model and evaluate it
        
        Args:
            name: Name of the model
            model: The ML model object
            X_train, y_train: Training data
            X_test, y_test: Testing data
        """
        print(f"\n🤖 Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ {name} trained!")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        
        # Store results
        self.trained_models[name] = model
        self.results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'model': model
        }
        
        return accuracy
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and compare them
        """
        print("\n" + "="*60)
        print("🚀 TRAINING ALL MODELS")
        print("="*60)
        
        accuracies = {}
        
        for name, model in self.models.items():
            accuracy = self.train_model(name, model, X_train, y_train, X_test, y_test)
            accuracies[name] = accuracy
        
        # Find best model
        best_model_name = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model_name]
        
        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON")
        print("="*60)
        
        for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            stars = "⭐" * int(acc * 10)
            print(f"{name:30s}: {acc*100:6.2f}% {stars}")
        
        print("\n" + "="*60)
        print(f"🥇 BEST MODEL: {best_model_name}")
        print(f"   Accuracy: {best_accuracy*100:.2f}%")
        print("="*60)
        
        return best_model_name, self.trained_models[best_model_name]
    
    def detailed_evaluation(self, model_name, X_test, y_test):
        """
        Show detailed evaluation of a model
        """
        print(f"\n📊 DETAILED EVALUATION: {model_name}")
        print("="*60)
        
        y_pred = self.results[model_name]['predictions']
        
        # Classification Report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Diabetes', 'Has Diabetes']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              No    Yes")
        print(f"Actual No  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
        print(f"Actual Yes [{cm[1][0]:4d}  {cm[1][1]:4d}]")
        
        print("\n💡 What this means:")
        print(f"   ✅ Correctly predicted NO diabetes: {cm[0][0]}")
        print(f"   ✅ Correctly predicted HAS diabetes: {cm[1][1]}")
        print(f"   ❌ Wrong predictions (False Positives): {cm[0][1]}")
        print(f"   ❌ Wrong predictions (False Negatives): {cm[1][0]}")
        
    def save_best_model(self, model_name, model, path='models/best_model.pkl'):
        """
        Save the best model for later use
        """
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, path)
        
        # Also save model info
        with open('models/model_info.txt', 'w') as f:
            f.write(f"Best Model: {model_name}\n")
            f.write(f"Accuracy: {self.results[model_name]['accuracy']*100:.2f}%\n")
        
        print(f"\n💾 Best model saved to {path}")
        print(f"   Model type: {model_name}")
    
    def feature_importance(self, model_name, model, feature_names):
        """
        Show which features are most important for prediction
        """
        print(f"\n🎯 FEATURE IMPORTANCE ({model_name})")
        print("="*60)
        
        # Only some models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nMost Important Features:")
            for i in range(min(10, len(feature_names))):
                idx = indices[i]
                bars = "█" * int(importances[idx] * 50)
                print(f"{i+1:2d}. {feature_names[idx]:25s}: {importances[idx]:.4f} {bars}")
        else:
            print("   Feature importance not available for this model type")


def train_pipeline():
    """
    Complete training pipeline
    """
    print("\n🏥 DIABETES PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Train all models
    best_model_name, best_model = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Detailed evaluation
    trainer.detailed_evaluation(best_model_name, X_test, y_test)
    
    # Feature importance
    trainer.feature_importance(best_model_name, best_model, X_train.columns)
    
    # Save best model
    trainer.save_best_model(best_model_name, best_model)
    
    print("\n🎉 MODEL TRAINING COMPLETE!")
    print("   You can now use the model to make predictions!")
    
    return best_model


# Main execution
if __name__ == "__main__":
    train_pipeline()
    