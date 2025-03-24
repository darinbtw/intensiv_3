import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
import re
warnings.filterwarnings('ignore')

class ReinforcementPriceModel:
    """
    Class for training and using CatBoost model to predict reinforcement prices
    """
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.target_column = 'Цена на арматуру'
        self.date_column = 'dt'
    
    def _clean_numeric_columns(self, df):
        """
        Clean numeric columns by converting strings with commas, K, M, B suffixes, 
        and percentage values to float values
        """
        # Loop through all columns that might be numeric
        for col in df.columns:
            # Skip the date column
            if col == self.date_column:
                continue
                
            # Check if column contains strings that need conversion
            if df[col].dtype == 'object':
                try:
                    # Function to convert various numeric formats
                    def convert_to_number(value):
                        if isinstance(value, (int, float)):
                            return value
                        
                        if pd.isna(value):
                            return np.nan
                            
                        # Convert to string and remove any quotes
                        value = str(value).strip('"\'')
                        
                        # Handle percentage values
                        if '%' in value:
                            # Remove percentage sign and convert to float/100
                            return float(value.replace('%', '').replace(',', '')) / 100
                        
                        # Remove any commas in the number
                        value = value.replace(',', '')
                        
                        # Check for suffixes (K, M, B)
                        if value.endswith('K') or value.endswith('k'):
                            return float(value[:-1]) * 1000
                        elif value.endswith('M') or value.endswith('m'):
                            return float(value[:-1]) * 1000000
                        elif value.endswith('B') or value.endswith('b'):
                            return float(value[:-1]) * 1000000000
                        else:
                            # Try to convert to float directly
                            return float(value)
                    
                    # Apply the conversion function to the column
                    df[col] = df[col].apply(convert_to_number)
                    
                except Exception as e:
                    # If conversion fails, print details for debugging
                    print(f"Warning: Could not convert column '{col}' to numeric. Error: {e}")
                    
                    # Try to identify problematic values
                    if df[col].dtype == 'object':
                        unique_values = df[col].dropna().unique()
                        if len(unique_values) < 20:  # Only print if there aren't too many values
                            print(f"Sample unique values in column '{col}': {unique_values[:5]}")
                    
                    # Keep the column as is (might be a categorical column)
                    pass
                    
        return df
        
    def _prepare_data(self, data, additional_data=None):
        """
        Prepare data for the model by:
        1. Converting dates to datetime
        2. Creating time-based features
        3. Merging with additional data if provided
        4. Handling missing values
        5. Creating lag features for time series
        """
        df = data.copy()
        
        # Clean numeric columns that might have commas, K, M, B suffixes, percentages
        df = self._clean_numeric_columns(df)
        
        # Ensure date is in datetime format
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        df = df.sort_values(by=[self.date_column])
        
        # Create date-based features
        df['year'] = df[self.date_column].dt.year
        df['month'] = df[self.date_column].dt.month
        df['week_of_year'] = df[self.date_column].dt.isocalendar().week
        df['day_of_week'] = df[self.date_column].dt.dayofweek
        df['quarter'] = df[self.date_column].dt.quarter
        
        # Add is_month_start, is_month_end, is_quarter_start, is_quarter_end
        df['is_month_start'] = df[self.date_column].dt.is_month_start.astype(int)
        df['is_month_end'] = df[self.date_column].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[self.date_column].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[self.date_column].dt.is_quarter_end.astype(int)
        
        # Create lag features (previous weeks prices)
        for lag in range(1, 13):  # Create lags up to 12 weeks
            df[f'price_lag_{lag}'] = df[self.target_column].shift(lag)
            
        # Create rolling mean features at different windows
        for window in [2, 4, 8, 12]:
            df[f'price_rolling_mean_{window}'] = df[self.target_column].rolling(window=window).mean()
            df[f'price_rolling_std_{window}'] = df[self.target_column].rolling(window=window).std()
            
        # Create price momentum features (percent change)
        for period in [1, 2, 4, 8]:
            df[f'price_pct_change_{period}'] = df[self.target_column].pct_change(periods=period)
        
        # If additional data is provided, merge it
        if additional_data is not None:
            additional_df = additional_data.copy()
            additional_df = self._clean_numeric_columns(additional_df)
            additional_df[self.date_column] = pd.to_datetime(additional_df[self.date_column])
            
            # Merge on date
            df = pd.merge(df, additional_df, on=self.date_column, how='left')
        
        # Handle missing values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Drop rows with NaN values (should only be initial rows with lag features)
        df = df.dropna()
        
        # Print column types to help with debugging
        print("DataFrame column types after preparation:")
        print(df.dtypes)
        
        # Check for any remaining object columns and print sample values
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            print(f"\nWarning: Found object columns after preparation: {list(obj_cols)}")
            for col in obj_cols:
                print(f"Sample values for '{col}': {df[col].dropna().unique()[:5]}")
        
        return df
    
    def _select_features(self, df):
        """
        Select features for training, excluding the target and date column
        """
        # Exclude date column and target column
        self.feature_columns = [col for col in df.columns if col != self.date_column and col != self.target_column]
        
        # Additional check: exclude any object columns (non-numeric)
        object_columns = df[self.feature_columns].select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            print(f"Warning: Excluding object columns from features: {list(object_columns)}")
            self.feature_columns = [col for col in self.feature_columns if col not in object_columns]
        
        return df[self.feature_columns], df[self.target_column]
    
    def train(self, train_data, additional_data=None, test_size=0.2, random_state=42):
        """
        Train the CatBoost model on the provided data
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            DataFrame containing historical price data
        additional_data : pandas.DataFrame, optional
            DataFrame containing additional features
        test_size : float, optional
            Proportion of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing training results and metrics
        """
        print("Preparing data for training...")
        df = self._prepare_data(train_data, additional_data)
        
        # Select features and target
        X, y = self._select_features(df)
        
        print(f"Feature columns selected for training: {self.feature_columns}")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Define CatBoost model
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=random_state,
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Train the model
        print("Training CatBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Evaluate the model
        val_predictions = self.model.predict(X_val)
        train_predictions = self.model.predict(X_train)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance(prettified=True)
        
        # Print metrics
        print("\nModel Training Results:")
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Validation MAE: {val_mae:.2f}")
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Validation RMSE: {val_rmse:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(y_val.values, label='Actual')
        plt.plot(val_predictions, label='Predicted')
        plt.title('Actual vs Predicted Prices (Validation Set)')
        plt.xlabel('Sample Index')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('model', exist_ok=True)
        plt.savefig('model/validation_results.png')
        plt.close()
        
        # Print top features
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'metrics': {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2
            },
            'feature_importance': feature_importance
        }
    
    def save_model(self, filepath='./model/reinforcement_price_model.pkl'):
        """
        Save the trained model to a file
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and feature columns
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath='./model/reinforcement_price_model.pkl'):
        """
        Load a trained model from a file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {filepath}")
        return True
    
    def predict_future(self, data, additional_data=None, weeks_ahead=4, current_date=None):
        """
        Predict prices for future weeks
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing historical price data
        additional_data : pandas.DataFrame, optional
            DataFrame containing additional features
        weeks_ahead : int, optional
            Number of weeks to predict ahead
        current_date : str or datetime, optional
            Current date to start predictions from
            
        Returns:
        --------
        dict
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Prepare historical data
        df = self._prepare_data(data, additional_data)
        
        # If current_date is not provided, use the last date in the data
        if current_date is None:
            current_date = df[self.date_column].max()
        else:
            current_date = pd.to_datetime(current_date)
        
        # Filter data up to current_date
        df = df[df[self.date_column] <= current_date]
        
        if df.empty:
            raise ValueError(f"No data available up to {current_date}")
        
        # Get the last row of data to use as a starting point for prediction
        last_row = df.iloc[-1:].copy()
        
        # Store predictions
        predictions = []
        dates = []
        
        # Make iterative predictions for each week
        current_df = last_row.copy()
        
        for week in range(1, weeks_ahead + 1):
            # Update the date for the next prediction
            next_date = current_date + timedelta(days=7 * week)
            dates.append(next_date)
            
            # Update date-related features
            current_df[self.date_column] = next_date
            current_df['year'] = next_date.year
            current_df['month'] = next_date.month
            current_df['week_of_year'] = next_date.isocalendar()[1]
            current_df['day_of_week'] = next_date.dayofweek
            current_df['quarter'] = (next_date.month - 1) // 3 + 1
            
            # Update is_month_start, is_month_end, is_quarter_start, is_quarter_end
            current_df['is_month_start'] = 1 if next_date.day == 1 else 0
            current_df['is_month_end'] = 1 if (next_date + timedelta(days=1)).day == 1 else 0
            current_df['is_quarter_start'] = 1 if next_date.month in [1, 4, 7, 10] and next_date.day == 1 else 0
            current_df['is_quarter_end'] = 1 if next_date.month in [3, 6, 9, 12] and (next_date + timedelta(days=1)).day == 1 else 0
            
            # Make prediction
            X_pred = current_df[self.feature_columns]
            price_pred = self.model.predict(X_pred)[0]
            predictions.append(price_pred)
            
            # Update the current_df for the next iteration
            current_df[self.target_column] = price_pred
            
            # Update lag features for next prediction
            for lag in range(12, 0, -1):
                if f'price_lag_{lag}' in current_df.columns:
                    if lag == 1:
                        current_df[f'price_lag_{lag}'] = current_df[self.target_column]
                    else:
                        current_df[f'price_lag_{lag}'] = current_df[f'price_lag_{lag-1}']
            
            # Here you would need to update other rolling features as well
            # For simplicity, we'll keep the values from the last known data point
        
        # Prepare final prediction results
        last_known_price = df.iloc[-1][self.target_column]
        prediction_data = []
        
        for i, (pred_date, pred_price) in enumerate(zip(dates, predictions)):
            prediction_data.append({
                "week": i + 1,
                "dt": pred_date.strftime('%Y-%m-%d'),
                "price": pred_price
            })
        
        avg_prediction = np.mean(predictions)
        price_change_pct = ((avg_prediction - last_known_price) / last_known_price) * 100
        
        # Generate recommendation based on price trend
        if price_change_pct > 5:
            recommendation = "HOLD: Prices are expected to rise significantly. Recommend a smaller tender now."
            confidence = "High" if price_change_pct > 10 else "Medium"
        elif price_change_pct < -5:
            recommendation = "BUY: Prices are expected to drop. Recommend a larger tender for the entire period."
            confidence = "High" if price_change_pct < -10 else "Medium"
        else:
            recommendation = "NEUTRAL: Prices are expected to remain stable. Proceed with standard tender volume."
            confidence = "Medium"
        
        return {
            "success": True,
            "average_prediction": avg_prediction,
            "last_price": last_known_price,
            "price_change_pct": price_change_pct,
            "recommendation": recommendation,
            "confidence": confidence,
            "weeks_ahead": weeks_ahead,
            "weekly_predictions": prediction_data
        }


def main():
    """
    Main function to train and save the model
    """
    # Load the data (paths should be adjusted based on actual file locations)
    try:
        print("Loading data...")
        train_data = pd.read_csv('./vladislav_work/processed_data.csv')
        
        # Print a sample of the data to help with debugging
        print("\nSample of training data:")
        print(train_data.head())
        print("\nData columns and types:")
        print(train_data.dtypes)
        print("\nColumns with object (string) type:")
        object_cols = train_data.select_dtypes(include=['object']).columns
        for col in object_cols:
            print(f"Sample values for '{col}': {train_data[col].dropna().unique()[:5]}")
        
        # Check if additional data exists
        additional_data = None
        if os.path.exists('data/additional_features.csv'):
            additional_data = pd.read_csv('data/additional_features.csv')
            print("Additional features data loaded.")
            
            # Print a sample of additional data
            print("\nSample of additional data:")
            print(additional_data.head())
        
        # Initialize and train the model
        model = ReinforcementPriceModel()
        results = model.train(train_data, additional_data)
        
        # Save the model
        model.save_model()
        
        print("\nModel training complete and saved!")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()