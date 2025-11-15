import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import io
import time

import sys
import traceback

st.set_page_config(layout="wide", page_title="Time Series Models Dashboard")



st.markdown("**Startup info (debug)**")
st.write("Python executable:", sys.executable)
st.write("Python version:", sys.version.splitlines()[0])



st.title("Time Series Dashboard — AR, ARIMA, LSTM, Transformer")
st.markdown("Demo dashboard to fit and compare 4 forecasting models on a public time series.")

# Add a manual run button so nothing heavy runs until clicked
if "run_models" not in st.session_state:
    st.session_state.run_models = False

if st.button("Run selected models"):
    st.session_state.run_models = True

if not st.session_state.run_models:
    st.info("Select models in the sidebar and press **Run selected models** to start. Check terminal for errors.")
    st.stop()




# Try import torch; if unavailable, disable deep learning models
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------------------
# Load example dataset
# ---------------------
@st.cache_data
def load_airpassengers():
    # Monthly international airline passengers 1949-1960
    d = sm.datasets.get_rdataset("AirPassengers").data
    # dataset has 'time' and 'value' columns (older statsmodels versions). Normalize to pandas Series
    if "value" in d.columns:
        s = pd.Series(d["value"].values, index=pd.date_range(start="1949-01", periods=len(d), freq="M"))
    else:
        # fallback: use statsmodels built-in dataset loader
        ap = sm.datasets.airpassengers.load_pandas().data
        s = ap['value']
        s.index = pd.date_range(start="1949-01", periods=len(s), freq="MS")
    s.name = "passengers"
    return s

series = load_airpassengers()

# ---------------------
# Sidebar controls
# ---------------------
st.sidebar.header("Experiment settings")
horizon = st.sidebar.number_input("Forecast horizon (periods)", min_value=1, max_value=36, value=12, step=1)
train_pct = st.sidebar.slider("Train fraction (%)", 50, 95, 80)
model_choice = st.sidebar.multiselect(
    "Models to run",
    options=["AR", "ARIMA", "LSTM (PyTorch)", "Transformer (PyTorch)"],
    default=["AR", "ARIMA"]
)
if not TORCH_AVAILABLE:
    st.sidebar.info("PyTorch not available — LSTM & Transformer disabled.")
    model_choice = [m for m in model_choice if "PyTorch" not in m]

# DL hyperparameters (only used if torch available and selected)
epochs = st.sidebar.number_input("DL epochs", min_value=1, max_value=100, value=10, step=1)
batch_size = st.sidebar.number_input("DL batch size", min_value=8, max_value=256, value=32, step=8)
lr = st.sidebar.number_input("DL learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f", step=1e-4)
hidden_size = st.sidebar.number_input("DL hidden size", min_value=4, max_value=256, value=32, step=4)

st.sidebar.markdown("Tip: reduce epochs and hidden size for faster runs in class.")

# ---------------------
# Show series
# ---------------------
st.subheader("Time series (AirPassengers sample)")
col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(8, 3))
    series.plot(ax=ax)
    ax.set_title("Monthly passengers")
    st.pyplot(fig)
with col2:
    st.write("Series summary")
    st.write(series.describe())

# Prepare train/test split
n_train = int(len(series) * train_pct / 100)
train_series = series.iloc[:n_train]
test_series = series.iloc[n_train : n_train + horizon]

st.write(f"Using {n_train} points for training and forecasting {horizon} periods.")

# ---------------------
# Helper: metrics & plot
# ---------------------
def evaluate_and_plot(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse
    return mae, rmse

def plot_forecast(series, train_end_idx, preds_dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax, label="observed", color="black")
    for name, preds in preds_dict.items():
        idx = pd.date_range(start=series.index[train_end_idx] + pd.offsets.MonthBegin(1), periods=len(preds), freq=series.index.freq)
        ax.plot(idx, preds, marker="o", label=name)
    ax.axvline(series.index[train_end_idx], color="gray", linestyle="--")
    ax.legend()
    ax.set_title("Forecasts vs Observed")
    return fig

# ---------------------
# AR model
# ---------------------
results = {}
if "AR" in model_choice:
    st.subheader("AR model")
    p = st.number_input("AR lag (p)", min_value=1, max_value=24, value=12, step=1, key="ar_p")
    try:
        ar = AutoReg(train_series, lags=p, old_names=False).fit()
        ar_forecast = ar.predict(start=n_train, end=n_train + horizon - 1, dynamic=False)
        # ensure length horizon
        ar_forecast = pd.Series(ar_forecast).values[:horizon]
        results["AR"] = ar_forecast
        mae, rmse = evaluate_and_plot(test_series.values[:horizon], ar_forecast, "AR")
        st.write(f"AR MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    except Exception as e:
        st.error(f"AR failed: {e}")

# ---------------------
# ARIMA model
# ---------------------
if "ARIMA" in model_choice:
    st.subheader("ARIMA model")
    p_arima = st.number_input("ARIMA p", min_value=0, max_value=5, value=2, step=1, key="arima_p")
    d_arima = st.number_input("ARIMA d", min_value=0, max_value=2, value=1, step=1, key="arima_d")
    q_arima = st.number_input("ARIMA q", min_value=0, max_value=5, value=0, step=1, key="arima_q")
    try:
        arima = ARIMA(train_series, order=(p_arima, d_arima, q_arima)).fit()
        arima_forecast = arima.forecast(steps=horizon)
        arima_forecast = pd.Series(arima_forecast).values[:horizon]
        results["ARIMA"] = arima_forecast
        mae, rmse = evaluate_and_plot(test_series.values[:horizon], arima_forecast, "ARIMA")
        st.write(f"ARIMA MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

# ---------------------
# PyTorch dataset / utils for DL models
# ---------------------
if TORCH_AVAILABLE and any("PyTorch" in m for m in model_choice):
    class SeqDataset(Dataset):
        def __init__(self, series_values, seq_len):
            self.x = []
            self.y = []
            for i in range(len(series_values) - seq_len):
                self.x.append(series_values[i : i + seq_len])
                self.y.append(series_values[i + seq_len])
            self.x = torch.tensor(np.array(self.x), dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
            self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    def train_dl_model(model, train_loader, val_loader, epochs, lr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for ep in range(epochs):
            model.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())
            # optional small validation step
        return model

    def forecast_sequence(model, seed_seq, horizon):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        preds = []
        seq = torch.tensor(seed_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
        for _ in range(horizon):
            with torch.no_grad():
                out = model(seq)  # (1,1,1)
                val = out.cpu().numpy().ravel()[0]
                preds.append(val)
                # roll sequence
                seq = torch.cat([seq[:, 1:, :], torch.tensor([[[val]]], dtype=torch.float32).to(device)], dim=1)
        return np.array(preds)

    # LSTM model
    if "LSTM (PyTorch)" in model_choice:
        st.subheader("LSTM (PyTorch)")
        seq_len = st.number_input("LSTM seq length", min_value=2, max_value=48, value=12, step=1, key="lstm_seq")
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out).unsqueeze(-1)

        try:
            values = series.values.astype(float)
            train_vals = values[:n_train]
            ds = SeqDataset(train_vals, seq_len)
            if len(ds) < 1:
                st.error("Not enough data for chosen sequence length.")
            else:
                dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True)
                model_lstm = LSTMModel(input_size=1, hidden_size=int(hidden_size))
                model_lstm = train_dl_model(model_lstm, dl, None, int(epochs), float(lr))
                seed = train_vals[-seq_len:]
                lstm_preds = forecast_sequence(model_lstm, seed, int(horizon))
                results["LSTM"] = lstm_preds
                mae, rmse = evaluate_and_plot(test_series.values[:horizon], lstm_preds, "LSTM")
                st.write(f"LSTM MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        except Exception as e:
            st.error(f"LSTM failed: {e}")

    # Transformer model
    if "Transformer (PyTorch)" in model_choice:
        st.subheader("Transformer (PyTorch)")
        seq_len_t = st.number_input("Transformer seq length", min_value=2, max_value=48, value=12, step=1, key="trans_seq")
        class SimpleTransformer(nn.Module):
            def __init__(self, input_size=1, d_model=32, nhead=4, num_layers=1):
                super().__init__()
                self.input_fc = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.out_fc = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.input_fc(x)  # (B, L, d_model)
                x = self.encoder(x)
                out = x[:, -1, :]
                return self.out_fc(out).unsqueeze(-1)

        try:
            values = series.values.astype(float)
            train_vals = values[:n_train]
            ds_t = SeqDataset(train_vals, seq_len_t)
            if len(ds_t) < 1:
                st.error("Not enough data for chosen sequence length.")
            else:
                dl_t = DataLoader(ds_t, batch_size=int(batch_size), shuffle=True)
                model_trans = SimpleTransformer(input_size=1, d_model=int(hidden_size), nhead=4, num_layers=1)
                model_trans = train_dl_model(model_trans, dl_t, None, int(epochs), float(lr))
                seed = train_vals[-seq_len_t:]
                trans_preds = forecast_sequence(model_trans, seed, int(horizon))
                results["Transformer"] = trans_preds
                mae, rmse = evaluate_and_plot(test_series.values[:horizon], trans_preds, "Transformer")
                st.write(f"Transformer MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        except Exception as e:
            st.error(f"Transformer failed: {e}")

# ---------------------
# Show combined plot and allow download
# ---------------------
if results:
    fig = plot_forecast(series, n_train - 1, results)
    st.pyplot(fig)

    # Prepare CSV for download: combine predictions into a table
    df_out = pd.DataFrame(index=pd.date_range(start=series.index[n_train] + pd.offsets.MonthBegin(1), periods=horizon, freq=series.index.freq))
    for k, v in results.items():
        df_out[k] = np.array(v).ravel()
    out_buf = io.BytesIO()
    df_out.to_csv(out_buf)
    out_buf.seek(0)
    st.download_button("Download forecasts CSV", data=out_buf, file_name="forecasts.csv")
else:
    st.info("Select at least one model and press run (models run automatically when selected).")

# ---------------------
# Final notes
# ---------------------
st.markdown(
    """
    Notes:
    - AR and ARIMA use statsmodels and are fast.
    - LSTM and Transformer use PyTorch; ensure PyTorch is installed. Training is minimal by default.
    - For classroom use, reduce epochs and hidden sizes for quick iterations.
    - To run: in terminal activate venv and `streamlit run 4_streamlit_timeseries.py`.
    """
)