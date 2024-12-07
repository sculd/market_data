import pandas as pd
from collections import defaultdict

def df_bar_to_dollar_bar(df):
    dollar_bar_rows = []
    dollar_bar_size = 20000
    cumulative_dollar_volume_per_symbol = defaultdict(float)
    prev_volume_per_symbol = defaultdict(float)
    prev_minute_per_symbol = {}

    for timestamp, row in df.iterrows():
        symbol = row['symbol']
        if symbol not in prev_minute_per_symbol or prev_minute_per_symbol[symbol] != timestamp:
            prev_minute_per_symbol[symbol] = timestamp
            prev_volume_per_symbol[symbol] = 0

        incremented_volume = row['volume'] - prev_volume_per_symbol[symbol]
        cumulative_dollar_volume_per_symbol[symbol] += incremented_volume
        prev_volume_per_symbol[symbol] = row['volume']
        if cumulative_dollar_volume_per_symbol[symbol] >= dollar_bar_size:
            dollar_bar_row = {
                'timestamp': row['ingestion_timestamp'],
                'symbol': symbol,
                'close': row['close'],
                'dollar_volume': cumulative_dollar_volume_per_symbol[symbol],
            }
            cumulative_dollar_volume_per_symbol[symbol] = 0
            dollar_bar_rows.append(dollar_bar_row)

    df_dolloar_bar = pd.DataFrame(dollar_bar_rows).set_index('timestamp')
    return df_dolloar_bar
