"""
Project 4 - CSCI 2400 - Regression

Name: Nolan Bessire
"""

import pandas as pd
import numpy as np  
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing(as_frame=True)
df = housing.data
df['target'] = housing.target


def get_most_important_feature(housing_df: pd.DataFrame) -> str:
    """
    Return a string with the largest absolute value for a coefficient 
    after being normalized
    """
    from sklearn.preprocessing import StandardScaler
    features = housing_df.drop(columns = ["target"])
    val = housing_df["target"]
    model = LinearRegression()
    scaler = StandardScaler()
    #normalize features before fitting them
    featuresScaled = scaler.fit_transform(features)
    #run the regression
    model.fit(featuresScaled, val)
    #build database with the coefficients and corresponding feature
    coefficients = pd.DataFrame({'Feature': features.columns, 'Coefficient': model.coef_})
    #find the maximum absolute value of a coefficient and return the name of that feature
    mostImportant = coefficients.loc[coefficients['Coefficient'].abs().idxmax(), 'Feature']
    return mostImportant

    

def get_r2_scores(housing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with a list of regression features and their corresponding 
    r2 scores
    """
    from sklearn.metrics import r2_score
    features = housing_df.drop(columns = ['target'])
    values = housing_df['target']
    r2_scores = {}
    #loop through all features to calculate their r2 scores
    for feature in features:
        numpyArray = np.array(features[feature])
        numpyArray = numpyArray.reshape(-1, 1)
        model = LinearRegression()
        #run regression using only the current feature
        model.fit(numpyArray, values)
        #calculate corresponding r2 score
        r2 = r2_score(values, model.predict(numpyArray))
        r2_scores[feature] = r2
    df = pd.DataFrame(list(r2_scores.items()), columns = ['feature', 'r2_score'])
    return df

#print(df.head())
#print(get_most_important_feature(df))
#print(get_r2_scores(df))

