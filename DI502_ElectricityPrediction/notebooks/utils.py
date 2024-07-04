import pandas as pd
import numpy as np
import joblib
import zipfile
import os
from sklearn.preprocessing import LabelEncoder
import holidays

def load_data():
    # Specify the ZIP file name
    zip_filename = "../dataset/filtered.zip"

    # Extract the model file from the ZIP archive
    with zipfile.ZipFile(zip_filename, "r") as archive:
        # Extract the model file (named "your_model.pkl" in this example)
        archive.extract("filtered.pkl")
        
    # Load the model
    df = joblib.load("filtered.pkl")  # Replace with "pickle.load" if you used pickle

    os.remove("filtered.pkl")

    return df

def break_datetime(df):
    df['timestamp']= pd.to_datetime(df['timestamp'])
    df[['year','weekofyear','dayofweek']]= np.uint16(df['timestamp'].dt.isocalendar())
    df['month']= np.uint8(df['timestamp'].dt.month)
    df['hour']= np.uint8(df['timestamp'].dt.hour)
    return df

def log_transformation(df):
    df['log_meter_reading']=np.log1p(df['meter_reading'])
    df['log_square_feet']=np.log1p(df['square_feet'])
    return df

def nan_weather_filler(df):
    df = break_datetime(df)
    air_temp_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['air_temperature'].transform('mean')
    df['air_temperature'].fillna(air_temp_df, inplace=True)

    dew_temp_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['dew_temperature'].transform('mean')
    df['dew_temperature'].fillna(dew_temp_df, inplace=True)

    cloud_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['cloud_coverage'].transform('mean')
    df['cloud_coverage'].fillna(cloud_df, inplace=True)

    sea_level_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['sea_level_pressure'].transform('mean')
    df['sea_level_pressure'].fillna(sea_level_df, inplace=True)

    precip_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['precip_depth_1_hr'].transform('mean')
    df['precip_depth_1_hr'].fillna(precip_df, inplace=True)

    wind_dir_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['wind_direction'].transform('mean')
    df['wind_direction'].fillna(wind_dir_df, inplace=True)

    wind_speed_df=df.groupby(['site_id', 'dayofweek', 'weekofyear'])['wind_speed'].transform('mean')
    df['wind_speed'].fillna(wind_speed_df, inplace=True)

    df['air_temperature'].fillna(df['air_temperature'].median(), inplace=True)
    df['dew_temperature'].fillna(df['dew_temperature'].median(), inplace=True)
    df['wind_direction'].fillna(df['wind_direction'].median(), inplace=True)
    df['wind_speed'].fillna(df['wind_speed'].median(), inplace=True)
    df['cloud_coverage'].fillna(df['cloud_coverage'].median(), inplace=True)
    df['sea_level_pressure'].fillna(df['sea_level_pressure'].median(), inplace=True)
    df['precip_depth_1_hr'].fillna(df['precip_depth_1_hr'].median(), inplace=True)

    return df

def label_encode(df,feature):
    label_enc= LabelEncoder()
    label_enc.fit(df[feature])
    df[feature+'_encoded']= label_enc.transform(df[feature])
    return df

def circular_encode(df, feature, max_value):
    df[feature+'_sin'] = np.sin(2 * np.pi * df[feature] / max_value)
    df[feature+'_cos']  = np.cos(2 * np.pi * df[feature] / max_value)
    return df


def save_model(model,model_name):
    zip_filename = "../models/"+model_name+".zip"

    # Create a ZIP file and add the model object to it
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as archive:
        # Save the model to a temporary file
        temp_model_filename = "temp_model.pkl"
        joblib.dump(model, temp_model_filename)
        
        # Add the temporary model file to the ZIP archive
        archive.write(temp_model_filename, arcname=model_name+".pkl")

    # Remove the temporary model file
    os.remove(temp_model_filename)
    

def load_model(model_name):
    # Specify the ZIP file name
    zip_filename = "../models/"+model_name+".zip"

    # Extract the model file from the ZIP archive
    with zipfile.ZipFile(zip_filename, "r") as archive:
        # Extract the model file (named "your_model.pkl" in this example)
        archive.extract(model_name+".pkl")
        
    # Load the model
    model = joblib.load(model_name+".pkl")  # Replace with "pickle.load" if you used pickle

    os.remove(model_name+".pkl")

    return model

def apply_holidays(df):
    # Load holiday data for England
    england_holidays = holidays.UnitedKingdom(years=range(2016, 2017))

    # Load holiday data for the United States
    us_holidays = holidays.UnitedStates(years=range(2016, 2017))

    # Initialize 'is_holiday' column with zeros
    df['is_holiday'] = 0

    # Create 'is_holiday_england' feature
    england_mask = df['site_id'] == 1
    df.loc[england_mask, 'is_holiday'] = df.loc[england_mask, 'timestamp'].apply(lambda x: 1 if x in england_holidays else 0)

    # Create 'is_holiday_us' feature
    us_mask = df['site_id'] == 6
    df.loc[us_mask, 'is_holiday'] = df.loc[us_mask, 'timestamp'].apply(lambda x: 1 if x in us_holidays else 0)

    return df
