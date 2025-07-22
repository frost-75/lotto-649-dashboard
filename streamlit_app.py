import streamlit as st
import pandas as pd
import numpy as np
import uuid
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------ CONFIGURATION ------------------
SHEET_NAME = "Lotto649"
LOG_SHEET = "PredictionLog"
MAX_PREDICTIONS = 3
COOLDOWN_HOURS = 24

# ------------------ GOOGLE SHEETS AUTH ------------------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open(SHEET_NAME).sheet1
log_sheet = client.open(SHEET_NAME).worksheet(LOG_SHEET)

# ------------------ LOAD DATA ------------------
data = pd.DataFrame(sheet.get_all_records())
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date']).sort_values('Date')

# ------------------ UI: INTRO ------------------
st.title("Lotto 6/49 Predictor 🎲")
st.markdown("""
Welcome to the **Lotto 6/49 Predictor**! Here's how it works:

- 🔄 You can generate predictions **up to 3 times per session**
- 🕓 You will be locked out for **24 hours** after reaching the limit
- 📊 Predictions are based on past data from our **Google Sheet database**
- 🏆 The last 3 official draws are displayed below
""")

# ------------------ ANTI-SPAM & SESSION LIMIT ------------------
if 'uuid' not in st.session_state:
    st.session_state.uuid = str(uuid.uuid4())
if 'count' not in st.session_state:
    st.session_state.count = 0

user_id = st.session_state.uuid
now = datetime.utcnow()

# Load log
log_df = pd.DataFrame(log_sheet.get_all_records())
log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')
recent_logs = log_df[log_df['user_id'] == user_id]

if not recent_logs.empty:
    last_try = recent_logs['timestamp'].max()
    if (now - last_try) < timedelta(hours=COOLDOWN_HOURS):
        st.warning("🚫 You've reached your daily prediction limit. Please try again after 24 hours.")
        st.stop()

# ------------------ SHOW LAST 3 DRAWS ------------------
st.subheader("📅 Last 3 Winning Draws")
latest = data.sort_values('Date', ascending=False).head(3)
for i, row in latest.iterrows():
    st.markdown(f"**{row['Date'].date()}** → 🎱 {row['num1']} {row['num2']} {row['num3']} {row['num4']} {row['num5']} {row['num6']} ⭐ Bonus: {row['bonus']}")

# ------------------ MODEL ------------------
def train_model(df):
    X_raw = df[['num1','num2','num3','num4','num5','num6']].values.astype(float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X, y = [], []
    for i in range(len(X_scaled) - 10):
        X.append(X_scaled[i:i+10].flatten())
        y.append(X_scaled[i+10])
    X, y = np.array(X), np.array(y)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(6, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    return model, scaler

# ------------------ GENERATE ------------------
st.subheader("🎯 Generate New Predictions")
if st.session_state.count < MAX_PREDICTIONS:
    if st.button("🔮 Generate Prediction Set"):
        model, scaler = train_model(data)
        recent = scaler.transform(data[['num1','num2','num3','num4','num5','num6']].values[-10:]).flatten().reshape(1, -1)

        results = []
        for _ in range(10):
            noise = np.random.normal(0, 0.01, recent.shape)
            pred_scaled = model.predict(recent + noise, verbose=0)[0]
            pred = scaler.inverse_transform([pred_scaled])[0]
            main = list(sorted(set(np.round(np.clip(pred, 1, 49)).astype(int))))[:6]
            while len(main) < 6:
                n = np.random.randint(1, 50)
                if n not in main:
                    main.append(n)
            bonus = np.random.randint(1, 50)
            while bonus in main:
                bonus = np.random.randint(1, 50)
            results.append(main + [bonus])

        for i, draw in enumerate(results):
            st.markdown(f"**Prediction {i+1}**: 🎱 {draw[:6]} ⭐ Bonus: {draw[6]}")

        # Log to Google Sheet
        log_sheet.append_row([user_id, now.isoformat()])
        st.session_state.count += 1
else:
    st.warning("⚠️ You've reached the max of 3 prediction sets for this session.")
