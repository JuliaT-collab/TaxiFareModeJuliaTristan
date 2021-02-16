# imports
from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preprocessing = ColumnTransformer([('time', pipe_time, time_cols),
                              ('distance', pipe_distance, dist_cols)]
                          )
        self.pipeline = Pipeline(steps=[('preprocessing', preprocessing),
                            ('regressor', LinearRegression())])

        return self

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df, test=False)
    # set X and y
    X = df.drop(columns=['fare_amount'])
    y=df['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
    # train

    pipeline = Trainer(X_train, y_train)
    # train
    pipeline.run()
    # evaluate
    rmse = pipeline.evaluate(X_test, y_test)

    print(rmse)
