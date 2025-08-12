import os
import pandas as pd
import numpy as np
import mplfinance as mpf
from scipy.signal import argrelextrema

RAW_CSV_DIR = "data/raw_csv"
OUTPUT_DIR = "data/candlestick_images"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_chart(df, label, name):
    label_dir = os.path.join(OUTPUT_DIR, label)
    ensure_dir(label_dir)
    mpf.plot(df, type='candle', style='charles', volume=False,
             savefig=os.path.join(label_dir, f"{name}.png"))

def detect_head_shoulders(df):
    prices = df['Close'].values
    maxima = argrelextrema(prices, np.greater, order=3)[0]

    for i in range(len(maxima) - 2):
        left, head, right = maxima[i], maxima[i+1], maxima[i+2]
        if prices[head] > prices[left] and prices[head] > prices[right]:
            if abs(prices[left] - prices[right]) / prices[head] < 0.15:
                yield max(0, left - 5), min(len(df), right + 5)

def detect_flag(df):
    window = 15
    for start in range(len(df) - window):
        sub = df.iloc[start:start+window]
        change = (sub['Close'].iloc[-1] - sub['Close'].iloc[0]) / sub['Close'].iloc[0]
        if 0.02 < abs(change) < 0.08:
            yield start, start+window

def main():
    for file in os.listdir(RAW_CSV_DIR):
        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(RAW_CSV_DIR, file))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Detect Head & Shoulders
        for start, end in detect_head_shoulders(df):
            save_chart(df.iloc[start:end], "Head & Shoulders", f"{file}_HS_{start}_{end}")

        # Detect Flag
        for start, end in detect_flag(df):
            save_chart(df.iloc[start:end], "Flag", f"{file}_Flag_{start}_{end}")

        # Save random "None" samples
        for start in range(0, len(df)-30, 40):
            save_chart(df.iloc[start:start+30], "None", f"{file}_None_{start}")

    print("✅ Auto-labeled charts created.")

if __name__ == "__main__":
    main()
