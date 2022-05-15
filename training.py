from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestRegressor
from pickle import dump
from pickle import load
from data_transformation import transform_data


training_directory='data/train/'
#We transform the data, explained in data_transformation.py
training_dataframe, patient_list=transform_data(training_directory)
training_dataframe.to_csv('mixed_dataframe_train')



#We create our training data, we separate the sepsis label
X_train=training_dataframe.drop('SepsisLabel', axis=1)
y_train= training_dataframe['SepsisLabel']






########################333
#We define scaler
scaler = StandardScaler()
#We fit scaler on the training dataset
scaler.fit(X_train)
#We scale the training dataset
X_train_scaled = scaler.transform(X_train)
#We impute for missing data using an Iterative imputer, that we train
imp = IterativeImputer(max_iter=4, random_state=0)
imp.fit(X_train_scaled)
X_train_scaled=imp.transform(X_train_scaled)

#We now define our prediction model, a Random Forest Regressor
model=RandomForestRegressor(max_depth=23)
model.fit(X_train_scaled, y_train)




########### Saving the model, scaler and imputer
dump(model, open('model.pkl', 'wb'))
dump(scaler, open('scaler.pkl', 'wb'))
dump(imp, open('imp.pkl', 'wb'))
################## loading them
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
imp= load(open('imp.pkl', 'rb'))
############################222


########### loading our model, scaler and imputer
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
imp= load(open('imp.pkl', 'rb'))


# transform the test dataset
X_test_scaled = scaler.transform(X_train)
X_test_scaled=imp.transform(X_test_scaled)
# make predictions on the test set
yhat = model.predict(X_test_scaled)
for indexa, yi in enumerate(yhat):
    if yi >= 0.5:
        yhat[indexa] = 1
    else:
        yhat[indexa] = 0
print(yhat)


#######################################################################
# evaluate accuracy
acc = accuracy_score(y_train, yhat)
print('Test Accuracy:', acc)
f1=f1_score(y_train, yhat, average='macro')
print ('F1 Score:', f1)
pref=precision_recall_fscore_support(y_train, yhat, average=None,labels=[0,1])
print ('pref:', pref)
