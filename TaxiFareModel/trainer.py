# imports

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
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import datetime
import os
from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib
from params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, MODEL_NAME, MODEL_VERSION, STORAGE_LOCATION

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "JuliaT-collab"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"



class Trainer():
    def __init__(self, X, y, experiment_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = experiment_name


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

        self.mlflow_log_param('model', 'LinearRegression')

        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""
        reg = self.run()
        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(reg, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        # Then do other things...
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename(filename='model.joblib')
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return rmse


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



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

    pipeline = Trainer(X_train, y_train, EXPERIMENT_NAME)
    # train
    pipeline.run()
    pipeline.save_model()
    # evaluate
    rmse = pipeline.evaluate(X_test, y_test)
    print(rmse)
