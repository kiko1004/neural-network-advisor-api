from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf

class Predictor:
    def __init__(
            self,
            df,
            class_prediction_string = 'class_prediction',
            binary_string = 'if_increase',
            ):
        self.df = df
        self.new_model = tf.keras.models.load_model(class_prediction_string)

        self.new_model_if_increase = tf.keras.models.load_model(binary_string)

        self._array = np.array(['1% - 2%', '2% - 3%', '3%+', 'decrease 1%-2%', 'decrease 2%-3%',
                           'decrease more that 3%', 'insignificant decrease > -0.1%',
                           'insignificant increase < 0.1%', 'less than 1% decrease',
                           'less than 1% increase'])

        self._array_if_increase = np.array([0, 1])

        self.features = ['Close_pct_change', 'Volume_daily_increase_past',
                    'Open_daily_increase_past', 'High_daily_increase_past',
                    'Low_daily_increase_past', 'Close_weekly_increase_past',
                    'Volume_weekly_increase_past', 'Open_weekly_increase_past',
                    'High_weekly_increase_past', 'Low_weekly_increase_past',
                    'High_Low_range', 'is_start_week', 'is_end_week', 'is_dividends',
                    'is_stock_splits', 'Month', 'Month_1', 'Month_2', 'Month_3', 'Month_4',
                    'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
                    'Month_11', 'Month_12']

    def start_week(self, day):
        start = day - timedelta(days=day.weekday())
        if day == start:
            return 1
        else:
            return 0

    def end_week(self, day):
        start = day - timedelta(days=day.weekday())
        end = start + timedelta(days=4)
        if day == end:
            return 1
        else:
            return 0

    def format_df(self, df_decade, features):
        df_decade["Price_weekly_increase_future"] = df_decade['Close'].pct_change(periods=-5)
        df_decade["Close_pct_change"] = df_decade['Close'].pct_change(periods=1)
        df_decade["Volume_daily_increase_past"] = df_decade['Volume'].pct_change(periods=1)
        df_decade["Open_daily_increase_past"] = df_decade['Open'].pct_change(periods=1)
        df_decade["High_daily_increase_past"] = df_decade['High'].pct_change(periods=1)
        df_decade["Low_daily_increase_past"] = df_decade['Low'].pct_change(periods=1)

        df_decade["Close_weekly_increase_past"] = df_decade['Close'].pct_change(periods=5)
        df_decade["Volume_weekly_increase_past"] = df_decade['Volume'].pct_change(periods=5)
        df_decade["Open_weekly_increase_past"] = df_decade['Open'].pct_change(periods=5)
        df_decade["High_weekly_increase_past"] = df_decade['High'].pct_change(periods=5)
        df_decade["Low_weekly_increase_past"] = df_decade['Low'].pct_change(periods=5)

        df_decade['High_Low_range'] = df_decade["High"] - df_decade["Low"]
        df_decade['is_start_week'] = df_decade['Date'].apply(self.start_week)
        df_decade['is_end_week'] = df_decade['Date'].apply(self.end_week)

        df_decade['is_dividends'] = np.where(df_decade['Dividends'] > 0, 1, 0)
        df_decade['is_stock_splits'] = np.where(df_decade['Stock Splits'] > 0, 1, 0)
        df_decade['Price_weekly_increase_future'] = df_decade['Price_weekly_increase_future'].apply(lambda x: -1 * x)
        df_decade['Month'] = df_decade["Date"].apply(lambda x: x.month)
        df_dummie_month = pd.get_dummies(df_decade['Month'])
        df_dummie_month.columns = [f"Month_{col}" for col in df_dummie_month]
        df_decade = df_decade.join(df_dummie_month)
        return df_decade[features]

    def predicit_to_series(self, model, X, _array):
        predictions = model(X)
        top_k_values, top_k_indices = tf.nn.top_k(predictions, k=1)
        string_predictions = [_array[i] for i in top_k_indices]
        probabilities = [float(i) for i in top_k_values]
        return string_predictions, probabilities

    def predict_latest_possible(self, model, df, _array, features):
        latest_date = self.format_df(df, features = features)[-1:]
        _class, probability = self.predicit_to_series(model, latest_date.values, _array = _array)
        return _class, probability

    def predict_next_week(self):
        binary_prediction, binary_confidence = self.predict_latest_possible(
            model=self.new_model_if_increase, df=self.df, _array=self._array_if_increase, features=self.features
        )
        binary_prediction, binary_confidence = binary_prediction[0], binary_confidence[0]

        interval_prediction, interval_confidence = self.predict_latest_possible(
            model=self.new_model, df=self.df, _array=self._array, features=self.features
        )
        interval_prediction, interval_confidence = interval_prediction[0], interval_confidence[0]

        return binary_prediction, binary_confidence, interval_prediction, interval_confidence


