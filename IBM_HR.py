from joblib import load
from sklearn.metrics import f1_score
import pandas as pd

X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv')

model = load('./model/model_hr.joblib')

y_model_pred = model.predict(X_test)
print("F1 Score: ", f1_score(y_model_pred, y_test, pos_label='No'))