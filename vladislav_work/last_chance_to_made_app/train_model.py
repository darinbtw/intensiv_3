import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

def load_data():
    # Загрузите свои данные
    train = pd.read_csv('./vladislav_work/last_chance_to_made_app/train.csv', parse_dates=['dt'])
    additional = pd.read_csv('vladislav_work/last_chance_to_made_app/data/processed_data.csv', parse_dates=['dt'])
    merged = train.merge(additional, on='dt', how='left')
    return merged

def create_target(df, max_n=6):
    prices = df['Цена на арматуру'].values
    targets = []
    for i in range(len(prices)-max_n):
        current_price = prices[i]
        future_prices = prices[i+1:i+1+max_n]
        optimal_n = 0
        for n in range(1, max_n+1):
            if current_price < np.mean(future_prices[:n]):
                optimal_n = n
            else:
                break
        targets.append(optimal_n if optimal_n > 0 else 1)
    targets += [1]*max_n
    df['target'] = targets[:len(df)]
    return df.dropna()

def create_features(df):
    df = df.sort_values('dt')
    for lag in [1, 2, 3, 4]:
        df[f'lag_{lag}'] = df['Цена на арматуру'].shift(lag)
    df['rolling_mean_4'] = df['Цена на арматуру'].rolling(4).mean()
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['week_of_year'] = df['dt'].dt.isocalendar().week
    return df.dropna()

def train_model():
    df = load_data()
    df = create_target(df)
    df = create_features(df)
    
    X = df.drop(['target', 'dt', 'Цена на арматуру'], axis=1)
    y = df['target']
    
    model = LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200)
    
    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
    
    joblib.dump(model, 'model.pkl')
    return model

if __name__ == '__main__':
    train_model()