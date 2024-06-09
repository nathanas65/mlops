from typing import Tuple
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(data, *args, **kwargs
) -> Tuple[DictVectorizer, LinearRegression]:
    categorical = ['PULocationID', 'DOLocationID']

    dv = DictVectorizer()

    #train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    train_dicts = data[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = data[target].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    # mean_squared_error(y_train, y_pred, squared=False)

    print(lr.intercept_)

    return dv, lr

