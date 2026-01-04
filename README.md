# 🏥 Diabetes Prediction System

An AI-powered health assessment tool that predicts diabetes risk using Machine Learning and Natural Language Processing.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![NLP](https://img.shields.io/badge/NLP-TextBlob-green.svg)

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies](#technologies)
- [Screenshots](#screenshots)
- [Disclaimer](#disclaimer)

---

## ✨ Features

### 🤖 Machine Learning
- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM
- **Automatic Model Selection**: Chooses the best performing model
- **Feature Engineering**: Creates intelligent derived features
- **High Accuracy**: Achieves ~75-80% prediction accuracy

 Natural Language Processing
- Text Analysis: Analyzes patient descriptions using NLP
- Sentiment Analysis: Detects emotional tone in patient input
- Keyword Extraction: Identifies symptoms, lifestyle factors, and family history
- Word Clouds: Visual representation of text data

 Interactive Web Interface
- **User-Friendly Design**: Built with Streamlit
- **Real-time Predictions**: Instant risk assessment
- **Beautiful Visualizations**: Gauge charts, radar plots, and heatmaps
- **Health Recommendations**: Personalized advice based on risk factors

---

 Project Structure

```
diabetes-prediction/
│
├── data/
│   ├── diabetes.csv              # Original dataset
│   ├── X_train.csv               # Training features (generated)
│   ├── X_test.csv                # Testing features (generated)
│   ├── y_train.csv               # Training labels (generated)
│   └── y_test.csv                # Testing labels (generated)
│
├── models/
│   ├── best_model.pkl            # Trained ML model (generated)
│   ├── scaler.pkl                # Feature scaler (generated)
│   └── model_info.txt            # Model metadata (generated)
│
├── src/
│   ├── data_preprocessing.py     # Data cleaning and preparation
│   ├── model_training.py         # Model training and evaluation
│   └── predict.py                # Prediction module
│
├── reports/                      # Generated prediction reports
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

 Installation

Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- VS Code (recommended)

 Step-by-Step Setup

1.Clone or Download the Project
   ```bash
   # Create project folder
   mkdir diabetes-prediction
   cd diabetes-prediction
   ```

2. Create Virtual Environment
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

 Usage

Step 1: Prepare the Data

```bash
python src/data_preprocessing.py
```

What it does:
- Loads the diabetes dataset
- Handles missing values
- Creates new features
- Splits data into training/testing sets
- Scales features
- Saves processed data

Expected Output:
```
🚀 STARTING DATA PREPROCESSING PIPELINE
📂 Loading data...
✅ Data loaded! Shape: (768, 9)
🔧 Handling missing values...
🎨 Creating new features...
✂️ Splitting data...
⚖️ Scaling features...
💾 Scaler saved to models/scaler.pkl
✅ PREPROCESSING COMPLETE!
```



 Step 2: Train the Model

```bash
python src/model_training.py
```

What it does:
- Trains 5 different ML models
- Compares their performance
- Saves the best model
- Shows detailed evaluation metrics

Expected Output:
```
🚀 TRAINING ALL MODELS
🤖 Training Logistic Regression...
✅ Logistic Regression trained! Accuracy: 77.27%
🤖 Training Decision Tree...
✅ Decision Tree trained! Accuracy: 73.38%
🤖 Training Random Forest...
✅ Random Forest trained! Accuracy: 78.57%
...
🥇 BEST MODEL: Random Forest
   Accuracy: 78.57%
💾 Best model saved to models/best_model.pkl
```

---

 Step 3: Make Predictions (CLI)

bash
python src/predict.py


Interactive Mode:
- Choose option 1 for single patient prediction
- Enter patient information when prompted
- Optionally describe symptoms for NLP analysis
- View comprehensive prediction report

Batch Mode:
- Choose option 2 for multiple patients
- Provide CSV file path
- Get predictions for all patients at once

---

 Step 4: Launch Web Application

```bash
streamlit run app.py
```

What happens:
- Opens a web browser automatically
- Shows beautiful interface at `http://localhost:8501`
- You can now use the app!

Web App Features:
1. Prediction Page
   - Enter patient data via sliders and inputs
   - Add text description for NLP analysis
   - Get instant predictions with visualizations
   - View personalized health recommendations

2.  Data Analysis Page
   - Explore dataset statistics
   - View feature distributions
   - Analyze correlations

3.  About Page
   - Project information
   - Model details
   - Technology stack

---

 Model Performance

### Model Comparison

| Model                  | Accuracy | Notes                          |
|------------------------|----------|--------------------------------|
| Random Forest          | 78.57%   | Best overall performance       |
| Gradient Boosting      | 77.92%   | Strong, close second           |
| Logistic Regression    | 77.27%   | Fast, interpretable            |
| SVM                    | 76.62%   | Good for small datasets        |
| Decision Tree          | 73.38%   | Simple, less accurate          |

### Performance Metrics

```
Classification Report (Random Forest):

              precision    recall  f1-score   support

No Diabetes       0.81      0.87      0.84       100
Has Diabetes      0.74      0.64      0.69        54

    accuracy                          0.79       154
```

### Key Insights
-  Model correctly identifies ~79% of cases
-  Better at detecting "No Diabetes" (87% recall)
-  More conservative with "Has Diabetes" (64% recall)
-  Balanced approach minimizes false negatives

---

Technologies

### Core Technologies
- Python 3.8+: Programming language
- NumPy & Pandas: Data manipulation
- Scikit-learn: Machine learning algorithms
 **Machine Learning Models**
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines

### NLP & Text Processing
- **TextBlob**: Sentiment analysis and text processing
- **WordCloud**: Text visualization

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts

### Web Framework
- **Streamlit**: Web application framework

---

## 🖼️ Screenshots

### 1. Data Preprocessing
```
🚀 STARTING DATA PREPROCESSING PIPELINE
====================================
📂 Loading data...
✅ Data loaded! Shape: (768, 9)
   (That means 768 patients, 9 features)

📊 Dataset Overview:
   Pregnancies  Glucose  BloodPressure  ...
0            6      148             72  ...
1            1       85             66  ...
```

### 2. Model Training
```
🏆 MODEL COMPARISON
====================================
Random Forest              : 78.57% ⭐⭐⭐⭐⭐⭐⭐⭐
Gradient Boosting          : 77.92% ⭐⭐⭐⭐⭐⭐⭐
Logistic Regression        : 77.27% ⭐⭐⭐⭐⭐⭐⭐
```

### 3. Web Interface
- Beautiful, modern UI with blue theme
- Gauge charts showing risk percentage
- Radar charts comparing patient to healthy ranges
- Word clouds for text analysis
- Color-coded risk indicators (🟢🟡🔴)

---

## 🧪 Dataset Information

### Source
**Pima Indians Diabetes Database**
- Originally from National Institute of Diabetes and Digestive and Kidney Diseases
- Purpose: Predict diabetes based on diagnostic measurements

### Features (8 total)

1. **Pregnancies**: Number of times pregnant (0-17)
2. **Glucose**: Plasma glucose concentration (mg/dL)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (μU/mL)
6. **BMI**: Body mass index (weight in kg/(height in m)²)
7. **DiabetesPedigreeFunction**: Genetic influence score
8. **Age**: Age in years (21-81)

### Target Variable
- **Outcome**: 0 (No Diabetes) or 1 (Has Diabetes)

### Dataset Statistics
- **Total Samples**: 768 patients
- **Diabetes Cases**: 268 (34.9%)
- **No Diabetes**: 500 (65.1%)

---

## 🎓 Learning Outcomes

This project teaches:

### For Beginners
- Setting up Python development environment
-  Working with CSV data
-  Basic machine learning concepts
-  Creating web applications

### For Intermediate
- Data preprocessing techniques
- Feature engineering
- Model comparison and selection
- NLP integration
- Interactive visualizations

### For Advanced
- Production-ready ML pipeline
- Model persistence (saving/loading)
- Web deployment with Streamlit
- Combining ML with NLP

---

## 🔧Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
# Solution: Activate virtual environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Then reinstall
pip install -r requirements.txt
```

**2. Model Files Missing**
```bash
# Solution: Run preprocessing and training first
python src/data_preprocessing.py
python src/model_training.py
```

**3. Port Already in Use (Streamlit)**
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

**4. CSV File Not Found**
```bash
# Solution: Ensure diabetes.csv is in data/ folder
# Check file path matches exactly
```

---

## 🚀 Advanced Usage

### Custom Dataset

To use your own diabetes dataset:

1. Format CSV with same columns as original
2. Replace `data/diabetes.csv`
3. Run preprocessing and training again

### Hyperparameter Tuning

Edit `src/model_training.py` to tune models:

```python
# Example: Tune Random Forest
'Random Forest': RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=10,          # Deeper trees
    min_samples_split=5,   # Min samples to split
    random_state=42
)
```

### Export Predictions

Predictions are automatically saved in `reports/` folder:
- Single predictions: `patient_report_TIMESTAMP.txt`
- Batch predictions: `batch_predictions.csv`

---

## 📈 Future Enhancements

Potential improvements:

- [ ] Deep Learning models (Neural Networks)
- [ ] More NLP features (Named Entity Recognition)
- [ ] Real-time data integration
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] API endpoints (REST/GraphQL)
- [ ] Patient history tracking
- [ ] Email report generation
- [ ] Integration with EHR systems

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork and modify for learning
- Experiment with different models
- Add new features
- Improve visualizations

---

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This diabetes prediction system is:
-  An **educational tool** for learning ML/AI
-  A **demonstration** of ML applications in healthcare
-  For **research and learning purposes only**

This system is **NOT**:
-  A replacement for professional medical advice
-  A diagnostic tool for clinical use
-  Approved by any medical regulatory authority
-  Suitable for making medical decisions

**Always consult qualified healthcare professionals** for:
- Medical diagnosis
- Treatment decisions
- Health concerns
- Medication advice

The creators and contributors are not responsible for any medical decisions made based on this tool's output.

---

##  Support

Having trouble? Try:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all steps were followed in order
4. Verify virtual environment is activated
5. Check that all files are in correct folders

---

##  License

This project is for educational purposes. Feel free to use and modify for learning.

---

##  Acknowledgments

- **Dataset**: Pima Indians Diabetes Database
- **Libraries**: Scikit-learn, Streamlit, Pandas, NumPy
- **Inspiration**: Healthcare AI applications
- **Purpose**: Making ML education accessible

---

##  Additional Resources

### Learn More About:
- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/)
- **Data Science**: [Pandas Documentation](https://pandas.pydata.org/)
- **NLP**: [TextBlob Guide](https://textblob.readthedocs.io/)
- **Web Apps**: [Streamlit Documentation](https://docs.streamlit.io/)

### Recommended Next Steps:
1. Try different ML algorithms
2. Experiment with feature engineering
3. Add more visualizations
4. Deploy to cloud (Streamlit Cloud is free!)
5. Build similar projects for other diseases

---

**Made with ❤️ for learners | Powered by AI | Educational Purpose Only**

---

*Last Updated: 2025*
