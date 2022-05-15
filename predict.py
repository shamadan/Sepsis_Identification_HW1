#!/usr/bin/python
import pandas as pd
from data_transformation import transform_data
from pickle import load
import sys

testing_directory=sys.argv[1]

#We transform the data, explained in data_transformation.py
testing_dataframe, patient_list=transform_data(testing_directory)

# testing_dataframe.to_csv('testing_dataframe.csv')

########### loading our model, scaler and imputer
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
imp= load(open('imp.pkl', 'rb'))


#We create our test data, without the label we need to predict
X=testing_dataframe.drop('SepsisLabel', axis=1)
#We don't use the actual label
y_test= testing_dataframe['SepsisLabel']

# transform the test dataset
X_test_scaled = scaler.transform(X)
X_test_scaled=imp.transform(X_test_scaled)
# make predictions on the test set
yhat = model.predict(X_test_scaled)
for indexa, yi in enumerate(yhat):
    if yi >= 0.5:
        yhat[indexa] = 1
    else:
        yhat[indexa] = 0

prediction=pd.DataFrame()
prediction['Id']=pd.Series(patient_list)
prediction['SepsisLabel']=pd.Series(yhat)
prediction.to_csv('prediction.csv')








