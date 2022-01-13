# Import all necessary libraries
 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
from sklearn.metrics import mean_absolute_error

def cross_validation():

    df = fetch_ml_ratings(variant='100k')

    train = df.sample(frac=0.8, random_state=7)
    val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
    test = df.drop(train.index.tolist()).drop(val.index.tolist())

    svd = SVD(lr=0.001, reg=0.005, n_epochs=500, n_factors=15, early_stopping=False,
            shuffle=False, min_rating=1, max_rating=5)

    svd.fit(X=train, X_val=val)

    pred = svd.predict(test)
    mae = mean_absolute_error(test['rating'], pred)

    print(f'Test MAE: {mae:.2f}')


def k_fold_cross_validation(k=5):
    df = fetch_ml_ratings(variant='100k')

    train = df.sample(frac=0.8, random_state=7)
    test = df.drop(train.index.tolist())
    
    fold_size = int(train.shape[0] / k)
    
    print(fold_size)

    svd = SVD(lr=0.001, reg=0.005, n_epochs=100, n_factors=15, early_stopping=True,
            shuffle=False, min_rating=1, max_rating=5)
    
    for i in range(k):
        val = train[fold_size*i: fold_size*(i+1)+1]
        train_ = train.drop(val.index.tolist())
        svd.fit(X=train_, X_val=val)

    pred = svd.predict(test)
    mae = mean_absolute_error(test['rating'], pred)

    print(f'Test MAE: {mae:.2f}')

cross_validation()

