# Bike Rental Demand Prediction using Azure AutoML

## Project Overview
This project demonstrates an end-to-end Automated Machine Learning (AutoML) solution for predicting bike rental demand using Azure Machine Learning. The workflow covers data preparation, model training with AutoML, deployment as a web service, and performance monitoring.

## Dataset
- **Source**: Bike sharing hourly data (e.g., from UCI ML Repository)
- **Features**: 
  - Temporal (season, year, month, hour, weekday)
  - Weather (temp, humidity, windspeed)
  - Holiday/working day flags
- **Target**: `cnt` - Count of total bike rentals

## Azure ML Workflow

### 1. Workspace Setup
- Created Azure ML workspace
- Configured compute resources (CPU/GPU clusters)

### 2. Data Preparation
```python
# Sample code for data preprocessing
from azureml.core import Dataset

# Register dataset
dataset = Dataset.Tabular.from_delimited_files(path)
dataset = dataset.register(workspace=ws, name='bike_rentals')
3. AutoML Training
Configured regression task with primary metric = Normalized Root Mean Squared Error

Enabled feature engineering

Set 30 iterations with 5-fold cross-validation

4. Model Deployment
Deployed best model as ACI web service

Enabled application insights for monitoring

Results
Metric	Value
Best Model	VotingEnsemble
NRMSE	0.32
R² Score	0.89
Training Time	45 min
How to Use
Clone this repository

Set up Azure ML workspace

Update config.json with your workspace details

Run python train.py to execute pipeline

API Consumption
python
import requests
import json

data = {
    "data": [{
        "season": 2,
        "hr": 15,
        "temp": 0.72,
        "hum": 0.43,
        "windspeed": 0.12
    }]
}

headers = {'Content-Type':'application/json', 'Authorization': f'Bearer {key}'}
response = requests.post(endpoint, json.dumps(data), headers=headers)
print(response.json())
