import joblib
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# --- 1. SETUP & CONFIGURATION ---

# Define constants from your training script
N_STEPS = 24  # Lookback window
N_FEATURES = 13 # Number of features after adding time features

# Create FastAPI app
app = FastAPI(
    title="Foresight Energy Load Forecasting API",
    description="A high-performance API for short-term energy load forecasting using a Bidirectional LSTM model.",
    version="1.0.0"
)

# --- 2. LOAD ARTIFACTS ---

# Define the PyTorch model class (must be identical to the one used for training)
class BidirectionalLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout):
        super(BidirectionalLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=dropout, bidirectional=True)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_weights(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        context_vector = self.dropout(context_vector)
        out = self.fc(context_vector)
        return out

# Load artifacts when the application starts
try:
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BidirectionalLSTMAttention(
        input_dim=N_FEATURES,
        hidden_dim=128,
        n_layers=2,
        output_dim=1,
        dropout=0.2
    ).to(device)
    model.load_state_dict(torch.load('artifacts/model.pth', map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # Load scaler and other artifacts
    scaler = joblib.load('artifacts/scaler.pkl')
    with open('artifacts/artifacts.json', 'r') as f:
        artifacts = json.load(f)
    FILL_VALUES = artifacts['fill_values']
    COLUMN_ORDER = artifacts['columns']
    print("✅ Scaler and other artifacts loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error loading artifacts: {e}. Make sure model.pth, scaler.pkl, and artifacts.json are in the 'artifacts' directory.")
    model = None # Prevent app from running if artifacts are missing

# --- 3. HELPER FUNCTIONS (PREPROCESSING) ---

def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical time features to the dataframe."""
    data_copy = data.copy()
    data_copy['hour'] = data_copy.index.hour
    data_copy['day_of_week'] = data_copy.index.dayofweek
    data_copy['month'] = data_copy.index.month
    data_copy['hour_sin'] = np.sin(2 * np.pi * data_copy['hour']/24.0)
    data_copy['hour_cos'] = np.cos(2 * np.pi * data_copy['hour']/24.0)
    data_copy['day_of_week_sin'] = np.sin(2 * np.pi * data_copy['day_of_week']/7.0)
    data_copy['day_of_week_cos'] = np.cos(2 * np.pi * data_copy['day_of_week']/7.0)
    data_copy['month_sin'] = np.sin(2 * np.pi * data_copy['month']/12.0)
    data_copy['month_cos'] = np.cos(2 * np.pi * data_copy['month']/12.0)
    return data_copy.drop(['hour', 'day_of_week', 'month'], axis=1)

def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """
    Applies the full preprocessing pipeline to new input data.
    """
    # 1. Ensure datetime index
    data['dt'] = pd.to_datetime(data['dt'])
    data = data.set_index('dt')

    # 2. Resample to hourly frequency
    data = data.resample('h').mean()
    
    # 3. Fill missing values
    data.fillna(FILL_VALUES, inplace=True)
    
    # 4. Add time features
    data = add_time_features(data)

    # 5. Ensure correct column order
    data = data[COLUMN_ORDER]

    # 6. Scale data
    scaled_data = scaler.transform(data)

    return scaled_data

# --- 4. PYDANTIC MODELS (API DATA STRUCTURE) ---

# --- In backend/main.py, replace the existing Pydantic models (starting from line 96) ---

# --- 4. PYDANTIC MODELS (API DATA STRUCTURE) ---

class DataPoint(BaseModel):
    dt: str = Field(..., example="2010-11-01 14:00:00", description="Timestamp in 'YYYY-MM-DD HH:MM:SS' format")
    Global_active_power: float = Field(..., example=1.39)
    Global_reactive_power: float = Field(..., example=0.088)
    Voltage: float = Field(..., example=233.91)
    Global_intensity: float = Field(..., example=5.8)
    Sub_metering_1: float = Field(..., example=0.0)
    Sub_metering_2: float = Field(..., example=0.0)
    Sub_metering_3: float = Field(..., example=0.0)

class PredictionRequest(BaseModel):
    data: List[DataPoint] = Field(..., description=f"A list of the last {N_STEPS} or more hours of data.")
    forecast_horizon: int = Field(default=12, gt=0, le=72, description="Number of hours to forecast (1-72).")

class ForecastPoint(BaseModel):
    timestamp: str = Field(..., example="2010-11-01 15:00:00")
    predicted_kw: float = Field(..., example=1.52)

class PredictionResponse(BaseModel):
    forecast: List[ForecastPoint]

# --- 5. API ENDPOINT (replace the entire /predict endpoint) ---

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    df = pd.DataFrame([dp.dict() for dp in request.data])
    if len(df) < N_STEPS:
        raise HTTPException(status_code=400, detail=f"Insufficient data. At least {N_STEPS} hours required.")

    try:
        processed_data = preprocess_input(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}")

    # --- Autoregressive Forecasting Loop ---
    model_input = processed_data[-N_STEPS:]
    predictions_scaled = []
    
    current_timestamp = pd.to_datetime(df['dt'].iloc[-1])

    with torch.no_grad():
        for _ in range(request.forecast_horizon):
            input_tensor = torch.from_numpy(model_input).float().unsqueeze(0).to(device)
            
            # Get prediction (shape: [1, 1])
            prediction = model(input_tensor).cpu().numpy()[0]
            predictions_scaled.append(prediction)

            # --- Create the next input row ---
            # Get last known values for exogenous features
            last_known_features = model_input[-1, 1:]
            
            # Create a new row with the prediction as the target feature
            new_row_features = np.concatenate([prediction, last_known_features]).reshape(1, -1)
            
            # Update the input sequence
            model_input = np.append(model_input[1:], new_row_features, axis=0)

    # Inverse transform all predictions at once
    predictions_scaled_array = np.array(predictions_scaled).reshape(-1, 1)
    dummy_features = np.zeros((len(predictions_scaled_array), N_FEATURES))
    dummy_features[:, 0] = predictions_scaled_array.flatten()
    predictions_inversed = scaler.inverse_transform(dummy_features)[:, 0]

    # Create timestamps for the forecast horizon
    forecast_timestamps = pd.date_range(start=current_timestamp + pd.Timedelta(hours=1), periods=request.forecast_horizon, freq='h')

    # Format the response
    forecast_result = [
        ForecastPoint(timestamp=ts.strftime('%Y-%m-%d %H:%M:%S'), predicted_kw=float(pred))
        for ts, pred in zip(forecast_timestamps, predictions_inversed)
    ]

    return PredictionResponse(forecast=forecast_result)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Foresight API is running. Go to /docs for the interactive API documentation."}