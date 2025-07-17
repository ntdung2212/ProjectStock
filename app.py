import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import pmdarima as pm
from datetime import timedelta
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
import lime.lime_tabular
import os
import requests
import base64
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from vnstock import Vnstock
from vnstock.explorer.misc.gold_price import sjc_gold_price, btmc_goldprice
from vnstock.explorer.misc.exchange_rate import *

st.set_page_config(layout="wide")
st.title("AI-Powered Stock Analyzing & Forecasting")
MODEL_SAVE_PATH = "saved_models"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3"

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(inputs.shape[-1])(x)
    return x + res

# Build Transformer Model
def build_transformer_model(window_size):
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype=tf.float32)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error")
    return model

# Build LSTM Model
def build_lstm_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error")
    return model

# Save trained models
def save_trained_model(model, dataset_name, model_type):
    if model_type == "Transformer":
        save_path = os.path.join(MODEL_SAVE_PATH, f"{dataset_name}_{model_type}.keras")
    else:
        save_path = os.path.join(MODEL_SAVE_PATH, f"{dataset_name}_{model_type}.h5")

    model.save(save_path, include_optimizer=False)
    st.success(f"{model_type} model trained and saved successfully!")

# Train Model Function
def train_model(model_type, data, window_size, epochs, dataset_name):
    with st.spinner(f"Training {model_type} model... This may take a while."):
        values = data.reshape(-1, 1)
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)

        X, y = [], []
        for i in range(len(values_scaled) - window_size):
            X.append(values_scaled[i:i+window_size])
            y.append(values_scaled[i+window_size])

        X, y = np.array(X), np.array(y)

        if model_type == "LSTM":
            model = build_lstm_model(window_size)
        else:
            model = build_transformer_model(window_size)

        model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
        save_trained_model(model, dataset_name, model_type)

    st.rerun()


# Predict Function -----------------------------------------------------------
#---------------------------------------------------------------------------------
def run_arima_forecast(df, horizon, col="Close"):
    series = df[col].values
    model = pm.auto_arima(series, seasonal=False, trace=False)
    return model.predict(n_periods=horizon)

def predict_lstm(model_path, df, steps, input_col="Close", input_len=100):
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy mô hình: {model_path}")
        return [np.nan] * steps
    model = load_model(model_path)
    data = df[input_col].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    last_data = data_scaled[-input_len:]
    preds = []
    for _ in range(steps):
        x = last_data[-input_len:].reshape((1, input_len, 1))
        pred_scaled = model.predict(x, verbose=0)[0, 0]
        preds.append(pred_scaled)
        last_data = np.append(last_data, pred_scaled)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def predict_transformer(model_path, df, steps, input_col="Close", input_len=100):
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy mô hình: {model_path}")
        return [np.nan] * steps
    model = load_model(model_path)
    data = df[input_col].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    last_data = data_scaled[-input_len:]
    preds = []
    for _ in range(steps):
        x = last_data[-input_len:].reshape((1, input_len, 1))
        pred_scaled = model.predict(x, verbose=0)[0, 0]
        preds.append(pred_scaled)
        last_data = np.append(last_data, pred_scaled)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

#XAI Visualization----------------------------------------------------------------
#---------------------------------------------------------------------------------
def visualize_attention_weights(model, input_sequence, layer_name="attention", model_name="Transformer"):
    """
    Hiển thị attention weights của mô hình Transformer dưới dạng heatmap trên Streamlit.
    """
    try:
        attention_layer = model.get_layer(layer_name)
    except ValueError:
        #st.warning(f"Không tìm thấy layer attention với tên '{layer_name}'.")
        return

    try:
        # Nếu là custom layer có thuộc tính attention_weights_
        attention_weights = attention_layer.attention_weights_
    except AttributeError:
        try:
            # Nếu là layer MultiHeadAttention chuẩn
            _, attention_weights = attention_layer(input_sequence, input_sequence, return_attention_scores=True)
        except Exception as e:
            st.warning(f"Không thể truy xuất attention weights: {e}")
            return

    attention_weights_np = attention_weights.numpy().squeeze()
    if attention_weights_np.ndim == 3:
        attention_weights_np = attention_weights_np.mean(axis=0)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(attention_weights_np, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(f"Attention Weights Heatmap - {model_name}")
    ax.set_xlabel("Input timestep")
    ax.set_ylabel("Attention focus")
    plt.tight_layout()
    return fig


def lime_explanation_timeseries(model, X_train, X_test, instance_idx=0, model_name="LSTM"):
    """
    Giải thích dự đoán của mô hình cho 1 instance cụ thể với LIME.
    """
    X_train_flat = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    def model_predict_wrapper(X_2d):
        X_3d = X_2d.reshape((X_2d.shape[0], X_2d.shape[1], 1))
        return model.predict(X_3d).flatten()

    feature_names = [f"t-{X_train.shape[1]-i}" for i in range(X_train.shape[1])]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=feature_names,
        mode='regression',
        verbose=False
    )

    explanation = explainer.explain_instance(
        data_row=X_test_flat[instance_idx],
        predict_fn=model_predict_wrapper,
        num_features=10
    )

    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"LIME Explanation - {model_name} instance {instance_idx}")
    plt.tight_layout()
    return fig

def explain_shap_kernel_safe(model, X_train, model_name="LSTM", sample_size=50, nsamples=100):
    """
    Giải thích mô hình LSTM/Transformer bằng SHAP KernelExplainer.

    model : keras.Model
    X_train : np.ndarray
        Dữ liệu train (n_samples, timesteps, 1).
    model_name : str
    sample_size : int
        Số mẫu lấy từ train set để giải thích.
    nsamples : int
        Số lượng mẫu ngẫu nhiên cho KernelExplainer.
    """
    X_train_flat = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    background_data = X_train_flat[:sample_size]

    def predict_fn(X_2d):
        X_3d = X_2d.reshape((X_2d.shape[0], X_2d.shape[1], 1))
        preds = model.predict(X_3d).flatten()
        return preds

    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values = explainer.shap_values(background_data, nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_names = [f"t-{X_train.shape[1]-i}" for i in range(X_train.shape[1])]
    shap.summary_plot(shap_values, background_data, feature_names=feature_names, show=False)
    fig = plt.gcf()
    return fig

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def model_comparison(df, horizon, model_name, dataname, input_col):
    if model_name == "ARIMA":
        preds = run_arima_forecast(df, horizon, col=input_col)
    elif model_name == "LSTM":
        model_path = f"saved_models/{dataname}_LSTM.h5"
        preds = predict_lstm(model_path, df, horizon, input_col)
    elif model_name == "Transformer":
        model_path = f"saved_models/{dataname}_Transformer.keras"
        preds = predict_transformer(model_path, df, horizon, input_col)

    true_values = df[input_col].values[-horizon:]
    
    # Ensure the lengths match
    preds_aligned = preds[:len(true_values)]

    # Convert to NumPy arrays
    true_values = np.array(true_values, dtype=np.float64)
    preds_aligned = np.array(preds_aligned, dtype=np.float64)

    # Remove NaN or Inf values from both arrays
    mask = ~np.isnan(true_values) & ~np.isnan(preds_aligned) & np.isfinite(true_values) & np.isfinite(preds_aligned)
    if np.sum(mask) == 0:
        st.warning(f"Model {model_name} has only NaN or Inf predictions, skipping comparison.")
        return np.nan, np.nan  # Return NaN to indicate failure

    true_values = true_values[mask]
    preds_aligned = preds_aligned[mask]

    # Scale values to prevent overflow (for large numbers like "Volume")
    scaler = StandardScaler()
    true_values_scaled = scaler.fit_transform(true_values.reshape(-1, 1)).flatten()
    preds_aligned_scaled = scaler.transform(preds_aligned.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(true_values_scaled, preds_aligned_scaled))
    mae = mean_absolute_error(true_values_scaled, preds_aligned_scaled)

    return rmse, mae

def visual_option(data, fig):
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "RSI (14)", "MACD", "OBV", "ADX", "Stochastic Oscillator"],
        default=["20-Day SMA"])

    latest_signals = []  # Cảnh báo

    for indicator in indicators:
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data['Date'], y=sma, mode='lines', name='SMA (20)'))

        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data['Date'], y=ema, mode='lines', name='EMA (20)'))

        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data['Date'], y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data['Date'], y=bb_lower, mode='lines', name='BB Lower'))

        elif indicator == "VWAP":
            vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data['Date'], y=vwap, mode='lines', name='VWAP'))

        elif indicator == "RSI (14)":
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            fig.add_trace(go.Scatter(x=data['Date'], y=rsi, mode='lines', name='RSI (14)'))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            if rsi.iloc[-1] > 70:
                latest_signals.append("⚠️ RSI indicates overbought conditions (RSI > 70)")
            elif rsi.iloc[-1] < 30:
                latest_signals.append("✅ RSI indicates oversold conditions (RSI < 30)")

        elif indicator == "MACD":
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            fig.add_trace(go.Scatter(x=data['Date'], y=macd, mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=data['Date'], y=signal, mode='lines', name='Signal'))

        elif indicator == "OBV":
            obv = [0]
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                    obv.append(obv[-1] + data['Volume'].iloc[i])
                elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                    obv.append(obv[-1] - data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            fig.add_trace(go.Scatter(x=data['Date'], y=obv, mode='lines', name='OBV'))

        elif indicator == "ADX":
            high = data['High']
            low = data['Low']
            close = data['Close']
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=14).mean()
            fig.add_trace(go.Scatter(x=data['Date'], y=adx, mode='lines', name='ADX'))

        elif indicator == "Stochastic Oscillator":
            low14 = data['Low'].rolling(window=14).min()
            high14 = data['High'].rolling(window=14).max()
            percent_k = 100 * (data['Close'] - low14) / (high14 - low14)
            percent_d = percent_k.rolling(window=3).mean()
            fig.add_trace(go.Scatter(x=data['Date'], y=percent_k, mode='lines', name='%K'))
            fig.add_trace(go.Scatter(x=data['Date'], y=percent_d, mode='lines', name='%D'))
            fig.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=20, line_dash="dot", line_color="green", annotation_text="Oversold")
            if percent_k.iloc[-1] > 80:
                latest_signals.append("⚠️ Stochastic %K indicates overbought (>80)")
            elif percent_k.iloc[-1] < 20:
                latest_signals.append("✅ Stochastic %K indicates oversold (<20)")

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    if latest_signals:
        for signal in latest_signals:
            st.info(signal)


def financial_analysis_with_ollama(income_statement_df, organ_name, report_period, headlines,price_data):
    """
    Gửi báo cáo kết quả kinh doanh cho AI để phân tích bằng tiếng Anh.

    Parameters:
    - income_statement_df (pd.DataFrame): Bảng báo cáo kết quả kinh doanh.
    - organ_name (str): Tên doanh nghiệp.
    - report_period (str): 'quarter' hoặc 'year'.

    Returns:
    - str: Phân tích của mô hình AI.
    """
    period_text = "recent quarters" if report_period == "quarter" else "recent years"
    if income_statement_df is not None and not income_statement_df.empty:
        income_text = income_statement_df.to_csv(index=False)
    else:
        income_text = "Không có dữ liệu"

    
    price_text = price_data.to_csv(index=False)

    prompt = (
        f"You are a professional financial analyst specializing in both technical and sentiment analysis. Always give user both sentiment and technical analysis\n\n"
        f" **News Sentiment Analysis**:\n"
        f"Analyze the following news headlines related to {organ_name}. Determine whether each headline is **positive, neutral, or negative** with clear reasoning and percentages:\n"
        f"{headlines}\n\n"
        f"If there is no headline, skip the News Sentiment Analysis\n"
        f" **Technical Analysis**:\n"
        f"Review the financial report of {organ_name} provided below.\n"
        f"Analyze trends in **revenue, gross profit, operating expenses, and net income over the {period_text}**.\n"
        f"Evaluate the company’s financial health, business efficiency, and potential for future growth.\n\n"
        f"**Financial History**\n**{income_text}\n\n"
        f"**Price data**\n {price_text}\n\n"
        f"If there is no financial history, just ignore and continue to analyze.\n\n"
        f"Write a professional summary with key observations, avoiding repetition of raw numbers. Remember to mention the name of the organization {organ_name} in the introduction. "
        f"Always pay attention to the time frame in the chart and focus your analysis on that time frame.\n"
        f"Focus on financial interpretation and business implications.\n"
        f"Based on both analyses, provide a **Buy / Hold / Sell recommendation** with clear reasoning and percentage for each option.\n\n"
        f"Always add a disclaimer like this at the end of your answer: **This is a financial analysis based on the provided information and does not constitute financial advice. Investment decisions should be made after thorough research and consultation with a qualified financial advisor.**"
    )


    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "No response from AI.")
        return f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"API Request Failed: {e}"

def capture_chart_screenshot(fig, filename="chart_screenshot.png"):
    """Chụp ảnh biểu đồ bằng Edge WebDriver"""
    html_path = "temp_chart.html"
    fig.write_html(html_path)

    edge_options = Options()
    edge_options.use_chromium = True
    edge_options.add_argument("--headless")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--window-size=1920,1080")

    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)
    driver.get("file://" + os.path.abspath(html_path))

    time.sleep(2)
    driver.save_screenshot(filename)
    driver.quit()
    
    return filename

def analyze_explainability_with_ollama(shap_path, lime_path, attn_path, model_name):
    """
    Sends SHAP, LIME, and Attention charts to the AI chatbot for analysis.
    """
    images = [shap_path, lime_path]
    if attn_path:
        images.append(attn_path)

    encoded_images = []
    for img in images:
        with open(img, "rb") as image_file:
            encoded_images.append(base64.b64encode(image_file.read()).decode("utf-8"))

    prompt = (
        f"You are a financial AI expert specializing in Explainable AI (XAI).\n\n"
        f" **SHAP Analysis**:\n"
        f"Analyze the SHAP chart provided. Explain how different time steps contribute to the model's decision.\n\n"
        f" **LIME Analysis**:\n"
        f"Explain the LIME chart, highlighting key features influencing the stock price prediction.\n\n"
        f"Don't ask user to do anything else. Keep the analyzation clear, simple and professional"
    )
    if attn_path:
        prompt += f" **Transformer Attention Weights**:\nExplain how attention is distributed across time steps in the Transformer model."

    payload = {
        "model": "gemma3",
        "messages": [
            {"role": "user", "content": prompt, "images": encoded_images}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    return response.json().get("message", {}).get("content", "No AI explanation received.")

def prediction_page():
    st.header("Forecasting & Model Interpretation")

    # --- Chọn nguồn dữ liệu và khởi tạo vnstock ---
    source_option = st.sidebar.selectbox("Select data source:", ["VCI", "TCBS"], index=0)
    stock_client = Vnstock().stock(symbol='VCI', source=source_option)

    # --- Lấy danh sách cổ phiếu ---
    stock_list_df = stock_client.listing.symbols_by_exchange()
    symbol_options = sorted(stock_list_df['symbol'].unique().tolist())
    selected_symbol = st.sidebar.selectbox("Select stock symbol from list:", symbol_options)
    organ_name = stock_list_df[stock_list_df["symbol"] == selected_symbol]["organ_name"].values[0]

    # --- Tìm kiếm nhanh bằng ô nhập ---
    manual_input = st.sidebar.text_input("Or enter stock symbol manually:").upper().strip()
    if manual_input:
        if manual_input in symbol_options:
            selected_symbol = manual_input
            st.sidebar.success(f"Found stock symbol: {manual_input}")
        else:
            st.sidebar.warning("⚠️ Stock symbol not found!")
        organ_name = stock_list_df[stock_list_df["symbol"] == selected_symbol]["organ_name"].values[0]

    # --- Chọn khoảng thời gian và interval ---
    start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2016-03-08"))
    end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("today"))
    interval = st.sidebar.selectbox("Interval:", ["1m", "5m", "15m", "30m", "1D", "1W", "1M"], index=4)

    # --- Lấy dữ liệu giá ---
    stock = Vnstock().stock(symbol=selected_symbol, source=source_option)
    try:
        data = stock.quote.history(start=str(start_date), end=str(end_date), interval=interval)
        if data.empty:
            st.warning("There is no data in this time frame!")
            return
    except Exception as e:
        st.error(f"Unable to get the data: {e}")
        return

    data.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values("Date", inplace=True)

    dataname = selected_symbol  # Dùng mã cổ phiếu làm tên dataset

    st.subheader(f"{selected_symbol} - {organ_name}")


    # Tùy chọn huấn luyện nếu người dùng muốn
    trained_models = ["ARIMA"]
    lstm_path = f"saved_models/{dataname}_LSTM.h5"
    trans_path = f"saved_models/{dataname}_Transformer.keras"
    if os.path.exists(lstm_path):
        trained_models.append("LSTM")
    if os.path.exists(trans_path):
        trained_models.append("Transformer")

    st.sidebar.subheader("Train Model")
    model_to_train = st.sidebar.radio("Select Model to Train:", ["LSTM", "Transformer"])
    epochs = st.sidebar.slider("Epochs:", 5, 50, 10)
    if epochs > 10:
        st.sidebar.warning("⚠️ Training may take a long time for more than 10 epochs.")
    window_size = 100
    if st.sidebar.button("Train Model"):
        train_model(model_to_train, data["Close"].values, window_size, epochs, dataname)

    input_col = st.sidebar.selectbox("Select financial characteristics:", ["Open", "High", "Low", "Close", "Volume"])
    forecast_horizon = st.sidebar.slider("Number of days to predict: ", 1, 30, 5)
    model_choice = st.sidebar.selectbox("Select model:", trained_models)

    if st.sidebar.button("Forecast"):
        with st.spinner("Processing model..."):
            if model_choice == "ARIMA":
                preds = run_arima_forecast(data, forecast_horizon, col=input_col)
            elif model_choice == "LSTM":
                model_file = lstm_path
                preds = predict_lstm(model_file, data, forecast_horizon, input_col)
            else:
                model_file = trans_path
                preds = predict_transformer(model_file, data, forecast_horizon, input_col)

            if np.isnan(preds).all():
                st.warning("Cannot forecast because no model found.")
                return

            last_30_days = data.iloc[-30:]
            last_date = data["Date"].iloc[-1]
            future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_horizon)
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": preds})

            st.subheader(f"{selected_symbol} - {organ_name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=last_30_days["Date"], y=last_30_days[input_col], name="Actual", mode='lines+markers'))
            fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], name="Forecast", line=dict(dash='dot'), mode='lines+markers'))
            fig.update_layout(hovermode="x unified", xaxis=dict(range=[last_30_days["Date"].min(), forecast_df["Date"].max()]))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download CSV", forecast_df.to_csv(index=False).encode(), file_name="forecast.csv")

        if model_choice in ["LSTM", "Transformer"] and os.path.exists(model_file):
            st.subheader("Explainable AI")

            with st.spinner("Generating Explainability Charts..."):
                model_loaded = load_model(model_file)
                fig_att = None
                data_array = data[input_col].values.reshape(-1, 1)
                attn_path = None

                try:
                    X_train_shap = np.stack([data_array[i: i + 100] for i in range(len(data_array) - 300, len(data_array) - 100)], axis=0)
                    fig_shap = explain_shap_kernel_safe(model_loaded, X_train_shap, model_name=model_choice, sample_size=30, nsamples=50)
                    shap_path = "shap_chart.png"
                    fig_shap.savefig(shap_path)
                except:
                    shap_path = None

                if model_choice == "Transformer":
                    try:
                        last_input_sequence = data[input_col].values[-100:].reshape(1, 100, 1)
                        fig_att = visualize_attention_weights(model_loaded, last_input_sequence, layer_name="multi_head_attention", model_name="Transformer")
                        attn_path = "attn_chart.png"
                        fig_att.savefig(attn_path)
                    except:
                        attn_path = None

                try:
                    X_train = np.stack([data_array[i: i + 100] for i in range(len(data_array) - 100 - 200, len(data_array) - 100 - 50)], axis=0)
                    X_test = np.stack([data_array[i: i + 100] for i in range(len(data_array) - 100 - 20, len(data_array) - 100)], axis=0)
                    fig_lime = lime_explanation_timeseries(model_loaded, X_train, X_test, instance_idx=0, model_name=model_choice)
                    lime_path = "lime_chart.png"
                    fig_lime.savefig(lime_path)
                except:
                    lime_path = None

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_shap)
            with col2:
                st.pyplot(fig_lime)
                if fig_att:
                    st.pyplot(fig_att)

            with st.spinner("Analyzing AI insights..."):
                ai_explanation = analyze_explainability_with_ollama(shap_path, lime_path, attn_path, model_choice)

            st.subheader("AI Explanation of XAI Charts")
            st.write(ai_explanation)

    if st.sidebar.button("Model Comparison"):
        results = []
        for m in trained_models:
            rmse, mae = model_comparison(data, min(forecast_horizon, 5), m, dataname, input_col)
            results.append({"Model": m, "RMSE": rmse, "MAE": mae})

        compare_df = pd.DataFrame(results)
        st.table(compare_df)



def stock_analysis():
    st.header("Market Analysis")

    # --- Chọn nguồn và khởi tạo vnstock ---
    source_option = st.sidebar.selectbox("Select data source:", ["VCI", "TCBS"], index=0)
    stock_client = Vnstock().stock(symbol='VCI', source=source_option)

    # --- Lấy danh sách cổ phiếu ---
    stock_list_df = stock_client.listing.symbols_by_exchange()
    symbol_options = sorted(stock_list_df['symbol'].unique().tolist())
    selected_symbol = st.sidebar.selectbox("Select stock symbol from list:", symbol_options)
    organ_name = stock_list_df[stock_list_df["symbol"] == selected_symbol]["organ_name"].values[0]
    # Tìm kiếm nhanh bằng ô nhập
    manual_input = st.sidebar.text_input("Or enter stock symbol manually:").upper().strip()
    if manual_input:
        if manual_input in symbol_options:
            selected_symbol = manual_input
            st.sidebar.success(f"Found stock symbol: {manual_input}")
        else:
            st.sidebar.warning("⚠️ Stock symbol not found!")
        organ_name = stock_list_df[stock_list_df["symbol"] == selected_symbol]["organ_name"].values[0]
    
    # --- Chọn khoảng thời gian và interval ---
    start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2025-01-01"))
    end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("today"))
    interval = st.sidebar.selectbox("Interval:", ["1m","5m","15m","30m","1D", "1W", "1M"], index=4)

    # --- Lấy dữ liệu giá ---
    stock = Vnstock().stock(symbol=selected_symbol, source=source_option)
    try:
        price_data = stock.quote.history(start=str(start_date), end=str(end_date), interval=interval)
        if price_data.empty:
            st.warning("There is no data in this time period.")
            return
    except Exception as e:
        st.error(f"Unable to get data: {e}")
        return

    price_data.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data.sort_values("Date", inplace=True)

    # --- Hiển thị biểu đồ nến ---
    st.subheader(f"{selected_symbol} - {organ_name}")
    fig = go.Figure(data=[go.Candlestick(
        x=price_data['Date'],
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name="Candlestick"
    )])
    visual_option(price_data, fig)

    # --- Nhập tiêu đề tin tức ---
    st.subheader("Enter News Headlines for Sentiment Analysis")
    user_headlines = st.text_area("Provide financial news headlines (one per line):")


    # --- AI Financial Statement Analysis ---
    st.subheader("Financial Report")
    # Tuỳ chọn kỳ báo cáo
    report_period_option = st.selectbox("Select reporting period:", ["Quarter", "Year"], index=0)
    report_period = report_period_option.lower()
    try:
        stock = Vnstock().stock(symbol=selected_symbol, source=source_option)
        income_df = stock.finance.income_statement(period=report_period, lang="en", dropna=True)
        
        if not income_df.empty:
            st.dataframe(income_df, use_container_width=True)
            if st.button("AI Financial Statement and Sentiment Analysis"):
                with st.spinner("Analyzing, please wait..."):
                    ai_response = financial_analysis_with_ollama(income_df, organ_name, report_period, user_headlines.strip() if user_headlines.strip() else "No news provided.",price_data)
                    st.subheader("AI Analysis Result")
                    st.write(ai_response)
        else:
            st.warning("No financial report to be found.")
    except Exception as e:
        st.error(f"Error: {e}")

def crypto():
    st.header("Market Analysis")

    crypto_symbol_options = ["BTC", "ETH", "USDT", "USDC", "BNB", "XRP", "ADA", "SOL", "DOGE"]
    selected_symbol = st.sidebar.selectbox("Select crypto symbol from list:", crypto_symbol_options)
    crypto_name_map = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "USDT": "Tether",
    "USDC": "USD Coin",
    "BNB": "BNB (Build and Build)",
    "XRP": "XRP (Ripple)",
    "ADA": "Cardano",
    "SOL": "Solana",
    "DOGE": "Dogecoin"
    }
    organ_name = crypto_name_map.get(selected_symbol, selected_symbol)
    manual_input = st.sidebar.text_input("Or enter crypto symbol manually:").upper().strip()
    if manual_input:
        if manual_input in crypto_symbol_options:
            selected_symbol = manual_input
            st.sidebar.success(f"Found stock symbol: {manual_input}")
        else:
            st.sidebar.warning("⚠️ Crypto symbol not found!")
    
    # --- Chọn khoảng thời gian và interval ---
    start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2025-01-01"))
    end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("today"))

    # --- Lấy dữ liệu giá ---
    crypto_client = Vnstock().crypto(symbol=selected_symbol, source="MSN")
    try:
        price_data = crypto_client.quote.history(start=str(start_date), end=str(end_date))
        if price_data.empty:
            st.warning("There is no data in this time period.")
            return
    except Exception as e:
        st.error(f"Unable to get data: {e}")
        return

    price_data.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data.sort_values("Date", inplace=True)

    # --- Hiển thị biểu đồ nến ---
    st.subheader(f"{selected_symbol} - {organ_name}")
    fig = go.Figure(data=[go.Candlestick(
        x=price_data['Date'],
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name="Candlestick"
    )])
    visual_option(price_data, fig)

    # --- Nhập tiêu đề tin tức ---
    st.subheader("Enter News Headlines for Sentiment Analysis")
    user_headlines = st.text_area("Provide news headlines (one per line):")


    # --- AI Financial Statement Analysis ---
    st.subheader("Financial Report")
    try:
        crypto_client = Vnstock().crypto(symbol=selected_symbol, source="MSN")
        if st.button("AI Analysis"):
            with st.spinner("Analyzing, please wait..."):
                ai_response = financial_analysis_with_ollama(income_statement_df=None, organ_name=organ_name, report_period=None, headlines=user_headlines.strip() if user_headlines.strip() else "No news provided.", price_data=price_data)
                st.subheader("AI Analysis Result")
                st.write(ai_response)
    except Exception as e:
        st.error(f"Error: {e}")

def world_index():
    st.header("Market Analysis")

    
    symbol_options = ["INX", "DJI", "COMP", "RUT", "NYA", "RUI", "RUA", "UKX", "DAX", "PX1", "N225", "000001", "HSI", "SENSEX", "ME00000000"]
    selected_symbol = st.sidebar.selectbox("Select symbol from list:", symbol_options)
    index_name_map = {
    "INX": "S&P 500 Index",
    "DJI": "Dow Jones Industrial Average",
    "COMP": "Nasdaq Composite Index",
    "RUT": "Russell 2000 Index",
    "NYA": "NYSE Composite Index",
    "RUI": "Russell 1000 Index",
    "RUA": "Russell 3000 Index",
    "UKX": "FTSE 100 Index",
    "DAX": "DAX Index",
    "PX1": "CAC 40 Index",
    "N225": "Nikkei 225 Index",
    "000001": "Shanghai SE Composite Index",
    "HSI": "Hang Seng Index",
    "SENSEX": "S&P BSE Sensex Index",
    "ME00000000": "S&P/BMV IPC"
    }
    organ_name = index_name_map.get(selected_symbol, selected_symbol)
    manual_input = st.sidebar.text_input("Or enter symbol manually:").upper().strip()
    if manual_input:
        if manual_input in symbol_options:
            selected_symbol = manual_input
            st.sidebar.success(f"Found stock symbol: {manual_input}")
        else:
            st.sidebar.warning("⚠️ International symbol not found!")
    
    # --- Chọn khoảng thời gian và interval ---
    start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2025-01-01"))
    end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("today"))

    # --- Lấy dữ liệu giá ---
    index = Vnstock().world_index(symbol=selected_symbol, source="MSN")
    try:
        price_data = index.quote.history(start=str(start_date), end=str(end_date))
        if price_data.empty:
            st.warning("There is no data in this time period.")
            return
    except Exception as e:
        st.error(f"Unable to get data: {e}")
        return

    price_data.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data.sort_values("Date", inplace=True)

    # --- Hiển thị biểu đồ nến ---
    st.subheader(f"{selected_symbol} - {organ_name}")
    fig = go.Figure(data=[go.Candlestick(
        x=price_data['Date'],
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name="Candlestick"
    )])
    visual_option(price_data, fig)

    # --- Nhập tiêu đề tin tức ---
    st.subheader("Enter News Headlines for Sentiment Analysis")
    user_headlines = st.text_area("Provide news headlines (one per line):")


    # --- AI Financial Statement Analysis ---
    st.subheader("Financial Report")
    try:
        index = Vnstock().world_index(symbol=selected_symbol, source="MSN")
        if st.button("AI Analysis"):
            with st.spinner("Analyzing, please wait..."):
                ai_response = financial_analysis_with_ollama(income_statement_df=None, organ_name=organ_name, report_period=None, headlines=user_headlines.strip() if user_headlines.strip() else "No news provided.", price_data=price_data)
                st.subheader("AI Analysis Result")
                st.write(ai_response)
    except Exception as e:
        st.error(f"Error: {e}")

def gold_forex():
    st.header("Gold and Currency Prices")

    # --- Vàng SJC  ---
    st.subheader("🌟 SJC Gold Price")
    sjc_df = sjc_gold_price()
    if not sjc_df.empty:
        sjc_row = sjc_df.iloc[0]
        st.metric("Buy Price", f"{int(sjc_row['buy_price']):,} VND")
        st.metric("Sell Price", f"{int(sjc_row['sell_price']):,} VND")
        st.caption(f"Date: {sjc_row['date']}")
    else:
        st.warning("Could not fetch SJC gold price data.")

    # --- Vàng BTMC (hiển thị bảng nhiều loại) ---
    st.subheader(":moneybag: Bảo Tín Minh Châu Gold Prices")
    btmc_df = btmc_goldprice()

    if not btmc_df.empty:
        btmc_df_display = btmc_df[["name", "buy_price", "sell_price", "time"]].copy()
        btmc_df_display = btmc_df_display.sort_values("time", ascending=False).drop_duplicates("name")
        btmc_df_display = btmc_df_display.reset_index(drop=True)

        # Định dạng giá
        btmc_df_display["buy_price"] = btmc_df_display["buy_price"].apply(lambda x: f"{int(x):,}")
        btmc_df_display["sell_price"] = btmc_df_display["sell_price"].apply(lambda x: f"{int(x):,}")

        # Đổi tên cột cho hiển thị
        btmc_df_display.rename(columns={
            "name": "Product",
            "buy_price": "Buying Price",
            "sell_price": "Selling Price",
            "time": "Last Updated"
        }, inplace=True)

        st.dataframe(btmc_df_display, use_container_width=True)
    else:
        st.warning("Could not fetch BTMC gold price data.")

    # --- Giá ngoại tệ ---
    st.subheader("🌐 Currency Exchange Rates (VCB)")
    today = datetime.date.today().strftime("%Y-%m-%d")
    forex_df = vcb_exchange_rate(date=today)
    forex_df.rename(columns={
        "currency_code": "Currency Code",
        "currency_name": "Currency Name",
        "buy _cash": "Cash Buying",
        "buy _transfer": "Telegraphic Buying",
        "sell": "Selling"
    }, inplace=True)
    search = st.text_input("Search currency:")
    filtered_df = forex_df[
        forex_df["Currency Code"].str.contains(search, case=False) |
        forex_df["Currency Name"].str.contains(search, case=False)
    ] if search else forex_df
    display_columns = ["Currency Code", "Currency Name", "Cash Buying", "Telegraphic Buying", "Selling"]
    st.dataframe(filtered_df[display_columns], use_container_width=True)

# App routing
page = st.sidebar.radio("Select function:", ["Stock Analysis", "Stock Forecasting","Gold and Forex","Crypto","World Index"])
if page == "Stock Forecasting":
    prediction_page()
elif page == "Stock Analysis":
    stock_analysis()
elif page == "Gold and Forex":
    gold_forex()
elif page == "Crypto":
    crypto()
elif page == "World Index":
    world_index()