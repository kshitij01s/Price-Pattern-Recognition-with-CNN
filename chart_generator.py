import os
import pandas as pd
import mplfinance as mpf

RAW_CSV_PATH ='data/raw_csv'
OUTPUT_DIR = "data/candlestick_images/train""data/candlestick_images/val"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_chart(df, output_path):
    mpf.plot(df, type='candle', style='charles', volume=False, savefig=output_path)

def main():
    ensure_dir(OUTPUT_DIR)
    # Example: generate images from labeled CSV windows
    labels_csv = "data/labels.csv"
    labels_df = pd.read_csv(labels_csv)
    labels_df = labels_df.dropna(subset=['label'])
    labels_df['label'] = labels_df['label'].astype(str)

    for _, row in labels_df.iterrows():
        file_path = os.path.join(RAW_CSV_PATH, row['file'])
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        start, end = int(row['start_idx']), int(row['end_idx'])
        window_df = df.iloc[start:end]

        label_dir = os.path.join(OUTPUT_DIR, str(row['label']))
        ensure_dir(label_dir)
        output_file = os.path.join(label_dir, f"{row['file'].split('.')[0]}_{start}_{end}.png")
        save_chart(window_df, output_file)

    print("Charts generated successfully.")


if __name__ == "__main__":
    main()
