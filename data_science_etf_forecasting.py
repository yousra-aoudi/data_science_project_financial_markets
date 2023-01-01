import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
This clean_data method includes additional code to detect and remove outliers for the 'Open' and 'Close' columns of 
the data. Outliers are detected using the interquartile range (IQR) method, which calculates the range between the first
and third quartiles of the data and uses this range to identify values that are outside of the expected range. 
In this case study, values that are more than 1.5 times the IQR below the first quartile or above the third quartile 
are considered to be outliers and are removed from the DataFrame.
"""


class DataCleaner:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        # Convert the Date column to a datetime data type. This will allow you to perform time-based operations on
        # the data.
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.df = pd.DataFrame(self.data)
        self.date = self.df['Date']
        self.open = self.df['Open']
        self.close = self.df['Close']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.adj_close = self.df['Adj Close']
        self.signal = self.df['Signal']
        self.df_zero = None
        self.df_neg = None
        self.df_outliers = None

        print('df statistics \n', self.df.describe())

    def clean_data(self):
        """Cleans and preprocesses the data from the specified DataSource.
        The clean_data method is intended to clean and preprocess the data from the specified data source."""
        # code to clean and preprocess data

        # Initialize empty dataframe for negative values
        self.df_neg = pd.DataFrame()

        # Initialize empty dataframe for outliers values
        self.df_outliers = pd.DataFrame()

        # Remove all rows in the dateset with negatives values and
        self.df.dropna(inplace=True)

        # Save rows with zero values, and negatives values
        for col in [self.open, self.close, self.adj_close, self.signal]:
            self.df_neg = self.df_neg.append(self.df[col < 0])
        print('df with columns < 0 & == NaN \n', self.df_neg)

        # Generate report with negatives values
        self.df_neg.to_csv('neg.csv')

        # Remove rows with zero values
        self.df_zero = self.df[(self.df == 0)]
        print('df with columns == 0 \n', self.df_zero)
        self.df = self.df[(self.df != 0).all(1)]

        # Remove rows with duplicate dates
        self.df = self.df.drop_duplicates(subset='Date', keep='first')

        # Sort the data by date
        self.df.sort_values(by='Date', inplace=True, ascending=True)

        # Reset the index of the DataFrame
        self.df.reset_index(inplace=True, drop=True)

        # Detect and remove outliers for the 'Open' and 'Close' and 'Adj Close ' columns
        for col in [self.open, self.close, self.adj_close, self.signal]:
            # Calculate the IQR
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1

            # Calculate the lower and upper bounds
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            # Remove rows with values outside the bounds
            self.df_outliers = self.df_outliers.append(self.df[(col < lower_bound) | (col > upper_bound)])
            self.df = self.df[(col > lower_bound) & (col < upper_bound)]

        print('df with outliers \n', self.df_outliers)
        print('df with outliers description \n', self.df_outliers.describe())

        # Generate report with outliers
        self.df_outliers.to_csv('outliers.csv')

        # Reset the index of the DataFrame
        self.df.reset_index(inplace=True, drop=True)

    def get_cleaned_data(self):
        """Returns the cleaned data as a Pandas DataFrame.
        This get_cleaned_data method simply returns the value of the df attribute, which should contain the cleaned
        financial data after the clean_data method has been called."""
        # Generate report with outliers
        self.df.to_csv('new_etf_sample_dataset.csv')
        print('df statistics after cleaning data \n', self.df.describe())

        return self.df


if __name__ == '__main__':
    clean = DataCleaner("etf_sample_dataset.csv")

    # Plot the Signal and Close prices
    clean.clean_data()
    clean.get_cleaned_data()

"""
To analyze the effectiveness of the signal in forecasting ETF prices, there are several metrics that can be considered. 
Some potential metrics to consider are:

Mean Absolute Error (MAE): This measures the average magnitude of the errors between the predicted and actual 
ETF prices. A lower MAE indicates a better fit of the model.

Mean Squared Error (MSE): This measures the average squared difference between the predicted and actual ETF prices. 
Like MAE, a lower MSE indicates a better fit of the model.

Root Mean Squared Error (RMSE): This is the square root of the MSE, and is commonly used to compare the error of 
different models. Like MAE and MSE, a lower RMSE indicates a better fit of the model.

R-squared: This measures the proportion of the variance in the ETF price that is explained by the model. 
A higher R-squared value indicates a better fit of the model.

Correlation: This measures the strength and direction of the relationship between the predicted and actual ETF prices. 
A high positive correlation indicates a strong positive relationship between the two variables, while a high negative 
correlation indicates a strong negative relationship.

"""


class SignalAnalysis:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        # Convert the Date column to a datetime data type. This will allow you to perform time-based operations on
        # the data.
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.df = pd.DataFrame(self.data)
        self.date = self.df['Date']
        self.open = self.df['Open']
        self.close = self.df['Close']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.adj_close = self.df['Adj Close']
        self.signal = self.df['Signal']
        self.mae = None
        self.mse = None
        self.mean_returns = None
        self.mean_signal_returns = None
        self.rmse = None
        self.std_returns = None
        self.std_signal_returns = None
        self.sharpe_ratio = None
        self.signal_sharpe_ratio = None
        self.correlation = None

        print('df \n', self.df.head())

    def calculate_returns(self):
        """Calculate the returns for each day in the data"""
        self.df['returns'] = self.df['Close'].pct_change()

    def calculate_signal_returns(self):
        """Calculate the returns for each day based on the signal"""
        self.df['signal_returns'] = self.df['Signal'].pct_change()

    def calculate_mean_returns(self):
        """Calculate the mean returns for the data and signal returns"""
        self.mean_returns = self.df['returns'].mean()
        print(f'Mean returns: {self.mean_returns:.2f}')
        self.mean_signal_returns = self.df['signal_returns'].mean()
        print(f'Mean signal returns: {self.mean_signal_returns:.2f}')

    def calculate_standard_deviation(self):
        """Calculate the standard deviation of the returns and signal returns"""
        self.std_returns = self.df['returns'].std()
        print(f'std returns: {self.std_returns:.2f}')
        self.std_signal_returns = self.df['signal_returns'].std()
        print(f'std signal returns: {self.std_signal_returns:.2f}')

    def calculate_sharpe_ratio(self):
        """Calculate the Sharpe ratio for the returns and signal returns"""
        self.sharpe_ratio = self.mean_returns / self.std_returns
        print(f'sharpe ratio: {self.sharpe_ratio:.2f}')
        self.signal_sharpe_ratio = self.mean_signal_returns / self.std_signal_returns
        print(f'signal sharpe ratio: {self.signal_sharpe_ratio:.4f}')

    def calculate_mean_absolute_error(self):
        """Calculate the mean absolute error between the Signal and the Close price."""
        self.mae = np.mean(np.abs(self.df['signal_returns'] - self.df['returns']))
        print(f'Mean Absolute Error: {self.mae:.4f}')

    def calculate_mean_squared_error(self):
        """Calculate the mean squared error between the Signal and the Close price."""
        self.mse = np.mean((self.df['signal_returns'] - self.df['returns']) ** 2)
        print(f'Mean Squared Error: {self.mse:.4f}')

    def calculate_root_mean_squared_error(self):
        """Calculate the signal's root mean squared error (RMSE)"""
        self.rmse = self.mse ** 0.5
        print(f'The root mean squared error of the signal is {self.rmse:.4f}')

    def calculate_correlation(self):
        """Calculate the signal's correlation with the returns."""
        self.correlation = self.df['signal_returns'].corr(self.df['returns'])
        print(f'The correlation between the signal and the returns is {self.correlation:.4f}')

    def run_analysis(self):
        self.calculate_returns()
        self.calculate_signal_returns()
        self.calculate_mean_returns()
        self.calculate_standard_deviation()
        self.calculate_sharpe_ratio()
        self.calculate_mean_absolute_error()
        self.calculate_mean_squared_error()
        self.calculate_root_mean_squared_error()
        self.calculate_correlation()

    def plot_signal_vs_close(self):
        """Plot the Signal and Close prices over time."""
        plt.plot(self.df['Date'], self.df['signal_returns'], label='Signal returns')
        plt.plot(self.df['Date'], self.df['returns'], label='returns')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def get_data_analysed(self):
        """Returns the cleaned data as a Pandas DataFrame.
        This get_cleaned_data method simply returns the value of the df attribute, which should contain the cleaned
        financial data after the clean_data method has been called."""
        # Generate report with outliers
        self.df.to_csv('etf_sample_dataset_analyzed.csv')
        print('df statistics after cleaning data \n', self.df.describe())

        return self.df


if __name__ == '__main__':
    analysis = SignalAnalysis('new_etf_sample_dataset.csv')

    # Plot the Signal and Close prices
    analysis.run_analysis()
    analysis.plot_signal_vs_close()
    analysis.get_data_analysed()
    """
    Result
    The signal appears to have a very low forecasting accuracy, as demonstrated by the high mean absolute error
    and mean squared error. 
    The returns based on the signal also have a much higher standard deviation compared to the returns of the ETF, 
    indicating that the signal is not very consistent in its predictions. 
    The Sharpe ratio, which is a measure of risk-adjusted returns, is also lower for the signal returns compared to the 
    ETF returns, indicating that the signal is not generating returns that are sufficient to justify the higher risk. 
    Overall, these results suggest that the signal is not effective in forecasting ETF price movements.
    """


class MLForecaster:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        # Convert the Date column to a datetime data type. This will allow you to perform time-based operations on
        # the data.
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        # Replace NaN values with the mean of the column
        self.df = pd.DataFrame(self.data)
        self.df = self.df.dropna()
        # Reshape the data
        self.X = self.df['signal_returns'].values.reshape(-1, 1)
        self.y = self.df['returns'].values.reshape(-1, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.lr_model = LinearRegression()
        self.lr_predictions = None
        self.lr_mae = None
        self.lr_mse = None
        self.lr_rmse = None
        self.dt_model = DecisionTreeRegressor()
        self.dt_predictions = None
        self.dt_mae = None
        self.dt_mse = None
        self.dt_rmse = None
        self.rf_model = RandomForestRegressor()
        self.rf_predictions = None
        self.rf_mae = None
        self.rf_mse = None
        self.rf_rmse = None

    def train_linear_regression(self):
        return self.lr_model.fit(self.X_train, self.y_train)

    def test_linear_regression(self):
        self.lr_predictions = self.lr_model.predict(self.X_test)
        self.lr_mae = mean_absolute_error(self.y_test, self.lr_predictions)
        self.lr_mse = mean_squared_error(self.y_test, self.lr_predictions)
        self.lr_rmse = np.sqrt(self.lr_mse)
        print(f'Linear Regression MAE: {self.lr_mae:.4f}')
        print(f'Linear Regression MSE: {self.lr_mse:.4f}')
        print(f'Linear Regression RMSE: {self.lr_rmse:.4f}')

    def train_decision_tree(self):
        return self.dt_model.fit(self.X_train, self.y_train)

    def test_decision_tree(self):
        self.dt_predictions = self.dt_model.predict(self.X_test)
        self.dt_mae = mean_absolute_error(self.y_test, self.dt_predictions)
        self.dt_mse = mean_squared_error(self.y_test, self.dt_predictions)
        self.dt_rmse = np.sqrt(self.dt_mse)
        print(f'Decision Tree MAE: {self.dt_mae:.4f}')
        print(f'Decision Tree MSE: {self.dt_mse:.4f}')
        print(f'Decision Tree RMSE: {self.dt_rmse:.4f}')

    def train_random_forest(self):
        return self.rf_model.fit(self.X_train, self.y_train)

    def test_random_forest(self):
        self.rf_predictions = self.rf_model.predict(self.X_test)
        self.rf_mae = mean_absolute_error(self.y_test, self.dt_predictions)
        self.rf_mse = mean_squared_error(self.y_test, self.dt_predictions)
        self.rf_rmse = np.sqrt(self.rf_mse)
        print(f'Random Forest MAE: {self.rf_mae:.4f}')
        print(f'Random Forest MSE: {self.rf_mse:.4f}')
        print(f'Random Forest RMSE: {self.rf_rmse:.4f}')


"""
Instantiate the MLForecaster class and pass in the data to be used for training and testing. 
"""
if __name__ == "__main__":
    # Instantiate the MLForecaster class with the data and target variable
    forecaster = MLForecaster('etf_sample_dataset_analyzed.csv')

    # Train and test a linear regression model
    forecaster.train_linear_regression()
    forecaster.test_linear_regression()

    # Train and test a decision tree model
    forecaster.train_decision_tree()
    forecaster.test_decision_tree()

    # Train and test a random forest model
    forecaster.train_random_forest()
    forecaster.test_random_forest()

"""
Result
It looks like the linear regression model performed the best, with the lowest mean absolute error (MAE), mean squared 
error (MSE) and root mean squared error (RMSE) among the three models. This suggests that the linear regression model 
may be the most effective at predicting the ETF price using the given data.
"""
