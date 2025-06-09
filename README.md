# Supply Chain Intelligence Platform

An AI-powered supply chain intelligence platform featuring demand forecasting, spoilage prediction, and ETA estimation capabilities.

## 🚀 Project Overview

This project delivers a comprehensive machine learning solution for supply chain optimization through three core AI features:

- **📈 Demand Forecasting**: 7-day recursive forecasting for product demand prediction
- **🛡️ Spoilage Prediction**: Binary classification to predict shipment spoilage risk
- **⏰ ETA Prediction**: Accurate estimation of vehicle arrival times

## 🏗️ Architecture

The project follows a structured ML pipeline:
```
Data Ingestion → Feature Engineering → Model Training → Evaluation → Production Deployment
```

Each component is designed for scalability and production readiness with proper model serialization and performance tracking.

## 🔧 Technologies Used

- **Machine Learning**: XGBoost, LightGBM, Random Forest
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Web Interface**: Streamlit
- **Model Persistence**: Pickle, JSON
- **Development**: Jupyter Notebooks, Python 3.x

## 📊 Model Performance

| Model | Algorithm | Key Metric | Performance |
|-------|-----------|------------|-------------|
| Demand Forecasting | XGBoost Regressor | MAE | 8.75 |
| Spoilage Prediction | LightGBM Classifier | ROC-AUC | 0.8738 (87.38%) |
| ETA Prediction | Random Forest Regressor | R² Score | 99.2% |

## 🚦 Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm
```

### Running the Application
```bash
# Clone the repository
git clone <your-repo-url>
cd supply-chain-intelligence

# Launch the Streamlit app
streamlit run app.py
```

### Training Models
Each notebook can be run independently:
```bash
jupyter notebook src/training/train_1.ipynb  # Demand Forecasting
jupyter notebook src/training/train_2.ipynb  # Spoilage Prediction
jupyter notebook src/training/train_3.ipynb  # ETA Prediction
```

## 🎯 Features

### Demand Forecasting
- **Advanced Feature Engineering**: Time-based features, lags (7, 14 days), rolling averages
- **Recursive Forecasting**: 7-day ahead predictions using previous forecasts
- **Output Format**: Structured JSON with SKU-location-demand mapping

### Spoilage Prediction
- **Feature Engineering**: Temperature interactions, polynomial features
- **Binary Classification**: Spoilage risk assessment (0/1)
- **Production Ready**: Model persistence with column structure validation

### ETA Prediction
- **Multi-feature Input**: Distance, vehicle type, weather conditions
- **High Accuracy**: Sub-hour prediction accuracy (MAE: 0.63 hours)
- **Real-time Inference**: Live predictions through web interface

## 🖥️ Web Interface

The Streamlit application provides:
- **Interactive Input Forms**: User-friendly data entry for each prediction type
- **Real-time Predictions**: Instant model inference with formatted results  
- **Visual Outputs**: Clear presentation of forecasts, risk levels, and confidence metrics
- **Model Performance**: Display of key metrics and model information

## 📈 Model Details

### Demand Forecasting (XGBoost)
- **Target**: Daily product demand by SKU and location
- **Features**: Historical sales, time features, lag variables, rolling statistics
- **Validation**: Time-based train/test split for realistic evaluation

### Spoilage Prediction (LightGBM)  
- **Target**: Binary spoilage flag (0: No spoilage, 1: Spoilage)
- **Features**: Transit conditions, temperature interactions, shipment details
- **Metrics**: 79% accuracy, 87.38% ROC-AUC on test set

### ETA Prediction (Random Forest)
- **Target**: Trip duration in hours
- **Features**: Distance, vehicle specifications, weather, route conditions
- **Performance**: 99.2% R² score, 0.63 hours MAE

