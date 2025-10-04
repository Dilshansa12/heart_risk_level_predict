import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv(r'C:\Users\User\Desktop\IM PR & ML\ML Couse\video 6\heart_risk_level_predictor\cardio_dataset.csv').values
#print(dataset)
#print(dataset.shape)
data=dataset[:,0:7]
target=dataset[:,7]
#print('data:',data)
#print('target:',target)


train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
model=LinearRegression()
model.fit(train_data,train_target)

print('coef:',model.coef_)
print('a0:',model.intercept_)
predicted_target=model.predict(test_data)
print('Actaul:',test_target[:10])
print('Predicted:',predicted_target[:10])

from sklearn.metrics import r2_score

r2=r2_score(test_target,predicted_target)
print('r2:',r2)

import joblib

joblib.dump(model,'heart-risk-reg-model.sav')