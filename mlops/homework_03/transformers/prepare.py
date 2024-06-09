import pandas as pd


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(data, *args, **kwargs):
    
    data.tpep_dropoff_datetime = pd.to_datetime(data.tpep_dropoff_datetime)
    data.tpep_pickup_datetime = pd.to_datetime(data.tpep_pickup_datetime)

    
    data['duration'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data.duration = data.duration.dt.total_seconds() / 60

    
    data = data[(data.duration >= 1) & (data.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)
    

    return data


