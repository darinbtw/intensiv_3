import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os

class XGBoostPriceModel:
    """XGBoost model for reinforcement price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.target_col = 'Цена на арматуру'
        self.model_path = 'model/reinforcement_price_model.pkl'
        self.scaler_path = 'model/scaler.pkl'
        self.feature_cols_path = 'model/feature_cols.pkl'
        
    def prepare_features(self, df, additional_df=None):
        """
        Prepare features for model training/prediction
        
        Args:
            df: DataFrame with historical price data
            additional_df: Optional DataFrame with additional features
            
        Returns:
            DataFrame with prepared features
        """
        # Make sure data is sorted by date
        df = df.sort_values('dt')
        
        # Create time-based features
        df['year'] = df['dt'].dt.year
        df['month'] = df['dt'].dt.month
        df['week'] = df['dt'].dt.isocalendar().week
        df['day_of_week'] = df['dt'].dt.dayofweek
        df['quarter'] = df['dt'].dt.quarter
        
        # Create lag features (previous prices)
        for lag in [1, 2, 3, 4, 8, 12]:
            df[f'price_lag_{lag}'] = df[self.target_col].shift(lag)
            
        # Create rolling mean features
        for window in [2, 4, 8, 12]:
            df[f'rolling_mean_{window}'] = df[self.target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[self.target_col].rolling(window=window).std()
            
        # Calculate price momentum (change over time)
        for window in [1, 2, 4]:
            df[f'momentum_{window}'] = df[self.target_col].diff(window)
        
        # Merge additional features if provided
        if additional_df is not None:
            additional_df = additional_df.sort_values('dt')
            
            # Merge on date
            df = pd.merge_asof(df, additional_df, on='dt', direction='nearest')
            
        # Drop rows with NaN values (created by lag and rolling features)
        df = df.dropna()
        
        return df
    
    def train(self, train_df, additional_df=None, test_size=0.2, random_state=42):
        """
        Train the XGBoost model
        
        Args:
            train_df: DataFrame with historical price data
            additional_df: Optional DataFrame with additional features
            test_size: Portion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        # Make sure 'dt' column is datetime
        train_df['dt'] = pd.to_datetime(train_df['dt'])
        if additional_df is not None:
            additional_df['dt'] = pd.to_datetime(additional_df['dt'])
            
        # Prepare features
        df = self.prepare_features(train_df, additional_df)
        
        # Determine feature columns (exclude target and date columns)
        self.feature_cols = [col for col in df.columns if col not in [self.target_col, 'dt']]
        
        # Split into features and target
        X = df[self.feature_cols]
        y = df[self.target_col]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define XGBoost parameters - fixed for compatibility
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 200,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Create and train model - FIXED to use eval_set correctly
        self.model = xgb.XGBRegressor(**params)
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        
        # Different versions of XGBoost handle evaluation differently
        try:
            # Try modern version first
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=25,
                verbose=False
            )
        except TypeError:
            # Fallback for older versions
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                early_stopping_rounds=25,
                verbose=False
            )
                
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Get the best iteration if available
        best_iteration = getattr(self.model, 'best_iteration', None)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(self.model, max_num_features=15)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save model and artifacts
        self.save_model()
        
        return {
            'success': True,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'best_iteration': best_iteration,
            'feature_importance': {
                feature: importance for feature, importance in 
                zip(self.feature_cols, self.model.feature_importances_)
            }
        }
    
    def predict(self, df, additional_df=None, weeks_ahead=4, current_date=None):
        """
        Make price predictions for future weeks
        
        Args:
            df: DataFrame with historical price data
            additional_df: Optional DataFrame with additional features
            weeks_ahead: Number of weeks to predict
            current_date: Optional date to start prediction from
            
        Returns:
            Dictionary with predictions and recommendation
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                return {
                    'success': False,
                    'message': 'Model not loaded'
                }
        
        # Ensure datetime format
        df['dt'] = pd.to_datetime(df['dt'])
        if additional_df is not None:
            additional_df['dt'] = pd.to_datetime(additional_df['dt'])
        
        # Set current date if not provided
        if current_date is None:
            current_date = df['dt'].max()
        else:
            current_date = pd.to_datetime(current_date)
            
        # Get last known price
        last_price_row = df[df['dt'] <= current_date].sort_values('dt', ascending=False).iloc[0]
        last_price = last_price_row[self.target_col]
        last_date = last_price_row['dt']
        
        # Prepare initial feature data
        prepared_df = self.prepare_features(df, additional_df)
        
        # Get latest row of data to use as a base for predictions
        latest_data = prepared_df[prepared_df['dt'] <= current_date].sort_values('dt', ascending=False).iloc[0].to_dict()
        
        # Store predictions
        predictions = []
        weekly_data = []
        
        # Predict for each week ahead
        current_features = latest_data.copy()
        
        for week in range(1, weeks_ahead + 1):
            # Update date for next prediction
            pred_date = last_date + timedelta(days=7 * week)
            
            # Update time features
            current_features['year'] = pred_date.year
            current_features['month'] = pred_date.month
            
            # Handle week number calculation safely for different pandas versions
            try:
                current_features['week'] = pred_date.isocalendar()[1]  # For pandas >= 1.1.0
            except (TypeError, AttributeError):
                current_features['week'] = pred_date.week  # For older pandas versions
                
            current_features['day_of_week'] = pred_date.dayofweek
            current_features['quarter'] = (pred_date.month - 1) // 3 + 1
            
            # Extract feature values in correct order
            X_pred = pd.DataFrame([current_features])[self.feature_cols]
            
            # Scale features
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make prediction
            predicted_price = self.model.predict(X_pred_scaled)[0]
            predictions.append(predicted_price)
            
            # Update features for next prediction
            # Shift lag features
            for lag in range(12, 0, -1):
                if f'price_lag_{lag}' in current_features and lag > 1:
                    current_features[f'price_lag_{lag}'] = current_features.get(f'price_lag_{lag-1}', last_price)
            
            # Update lag_1 with current prediction
            if 'price_lag_1' in current_features:
                current_features['price_lag_1'] = predicted_price
                
            # Update rolling means (simplified)
            for window in [2, 4, 8, 12]:
                if f'rolling_mean_{window}' in current_features:
                    # Simple approximation of rolling average update
                    prev_mean = current_features[f'rolling_mean_{window}']
                    current_features[f'rolling_mean_{window}'] = (prev_mean * (window-1) + predicted_price) / window
            
            # Update momentum features
            for window in [1, 2, 4]:
                if f'momentum_{window}' in current_features:
                    if week > window:
                        current_features[f'momentum_{window}'] = predictions[-1] - predictions[-1-window]
                    else:
                        # Approximation for early predictions
                        current_features[f'momentum_{window}'] = predicted_price - last_price
            
            # Add to weekly data
            weekly_data.append({
                'week': week,
                'dt': pred_date.strftime('%Y-%m-%d'),
                'Цена на арматуру': predicted_price
            })
        
        # Calculate average prediction
        avg_prediction = np.mean(predictions)
        
        # Calculate price change
        price_change_pct = ((avg_prediction - last_price) / last_price) * 100
        
        # Generate recommendation
        if price_change_pct > 5:
            recommendation = "HOLD: Prices are expected to rise significantly. Recommend a smaller tender now."
            confidence = "High"
        elif price_change_pct < -5:
            recommendation = "BUY: Prices are expected to drop. Recommend a larger tender for the entire period."
            confidence = "High"
        elif price_change_pct > 2:
            recommendation = "NEUTRAL-HOLD: Prices are expected to rise slightly. Consider standard tender with slight reduction."
            confidence = "Medium"
        elif price_change_pct < -2:
            recommendation = "NEUTRAL-BUY: Prices are expected to drop slightly. Consider standard tender with slight increase."
            confidence = "Medium"
        else:
            recommendation = "NEUTRAL: Prices are expected to remain stable. Proceed with standard tender volume."
            confidence = "Medium"
        
        # Check for trend patterns
        price_trend = np.array(predictions)
        price_diff = np.diff(price_trend)
        
        # Check if consistently rising or falling
        if np.all(price_diff > 0):
            confidence = "High"
            recommendation = "WAIT: Prices consistently rising each week. Delay purchases if possible."
        elif np.all(price_diff < 0):
            confidence = "High"
            recommendation = "DELAY: Prices consistently falling each week. Consider delaying major purchases."
        
        # Check for volatility
        volatility = np.std(predictions) / np.mean(predictions) * 100
        if volatility > 5:
            confidence = "Low"
            recommendation += " High volatility detected, consider hedging strategies."
        
        return {
            'success': True,
            'average_prediction': avg_prediction,
            'last_price': last_price,
            'price_change_pct': price_change_pct,
            'recommendation': recommendation,
            'confidence': confidence,
            'weeks_ahead': weeks_ahead,
            'weekly_predictions': weekly_data,
            'volatility': volatility,
            'prediction_date': current_date.strftime('%Y-%m-%d')
        }
    
    def save_model(self):
        """Save the model and related components to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save feature columns
        joblib.dump(self.feature_cols, self.feature_cols_path)
        
        return True
    
    def load_model(self):
        """Load the model and related components from disk"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            
            # Load feature columns
            if os.path.exists(self.feature_cols_path):
                self.feature_cols = joblib.load(self.feature_cols_path)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# Example usage for integration with the main application
if __name__ == "__main__":
    # Sample data generation for testing
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='W')
    price_trend = np.linspace(40000, 60000, len(dates))
    
    # Add some seasonality and noise
    seasonality = 5000 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
    noise = np.random.normal(0, 1000, len(dates))
    
    prices = price_trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'dt': dates,
        'Цена на арматуру': prices
    })
    
    # Create additional features
    additional_df = pd.DataFrame({
        'dt': dates,
        'USD_Rate': np.random.uniform(70, 85, len(dates)),
        'Oil_Price': np.random.uniform(60, 120, len(dates)),
        'Steel_Index': np.random.uniform(100, 150, len(dates))
    })
    
    # Initialize model
    model = XGBoostPriceModel()
    
    print("Starting model training...")
    
    # Train model with error handling
    try:
        train_result = model.train(df, additional_df)
        print(f"Training results: MAE: {train_result['mae']:.2f}, RMSE: {train_result['rmse']:.2f}, R²: {train_result['r2']:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("Making predictions...")
    
    # Make predictions with error handling
    try:
        prediction_result = model.predict(df, additional_df, weeks_ahead=8)
        
        print(f"Prediction: {prediction_result['recommendation']}")
        print(f"Confidence: {prediction_result['confidence']}")
        print(f"Price change: {prediction_result['price_change_pct']:.2f}%")
        
        # Print weekly predictions
        for week in prediction_result['weekly_predictions']:
            print(f"Week {week['week']}: {week['dt']} - Price: {week['Цена на арматуру']:.2f}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(df['dt'], df['Цена на арматуру'], 'b-', label='Historical Data')
        
        # Plot predictions
        pred_dates = [pd.to_datetime(week['dt']) for week in prediction_result['weekly_predictions']]
        pred_prices = [week['Цена на арматуру'] for week in prediction_result['weekly_predictions']]
        plt.plot(pred_dates, pred_prices, 'r-o', label='Predictions')
        
        plt.title('Reinforcement Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()