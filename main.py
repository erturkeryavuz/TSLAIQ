import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils.indicators import add_technical_indicators

DATA_PATH = "data/tsla_data.csv"
TICKER = "TSLA"
START_DATE = "2010-06-29"
END_DATE = datetime.today().strftime("%Y-%m-%d")

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)
    return df

def main():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH):
        print("📂 Mevcut veri bulundu. Güncelleniyor...")

        old_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        last_date = old_df["Date"].max() + timedelta(days=1)
        today = datetime.today().date()

        if last_date.date() > today:
            print("✅ Veri zaten güncel.")
            df = old_df
        else:
            print(f"⬇️ Yeni veriler çekiliyor: {last_date.date()} ➡ {today}")
            new_df = get_stock_data(TICKER, last_date.strftime("%Y-%m-%d"), END_DATE)

            if not new_df.empty:
                new_df = add_technical_indicators(new_df)
                df = pd.concat([old_df, new_df], ignore_index=True)
                df.to_csv(DATA_PATH, index=False)
                print("✅ Yeni veriler eklendi ve CSV güncellendi.")
            else:
                df = old_df
                print("⚠️ Yeni veri bulunamadı.")
    else:
        print("⬇️ İlk kez veri çekiliyor...")
        df = get_stock_data(TICKER, START_DATE, END_DATE)
        df = add_technical_indicators(df)
        df.to_csv(DATA_PATH, index=False)
        print("✅ Veri kaydedildi: data/tsla_data.csv")

    print(df.tail())

if __name__ == "__main__":
    main()
