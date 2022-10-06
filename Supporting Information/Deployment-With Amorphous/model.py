import pandas as pd
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import StratifiedKFold
# Load the csv file
df = pd.read_excel("Final-A.xlsx")

X_OS = df[['Ge/Si', 'Al/T', 'OH/T', 'H2O/T', 'F/T', 'SDA/T', "B/T", "Na2O/T", "Cl/T", "Temperature", "time", "AR", "Area", "C/N", "rpm"]]
Y_OS= df['RC2']
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in cv.split(X_OS, Y_OS):
    X_Train, X_Test= X_OS.iloc[train_index], X_OS.iloc[test_index]
    Y_Train, Y_Test= Y_OS[train_index], Y_OS[test_index]

XGB = XGBClassifier(random_state=1, n_estimators=1600, learning_rate=1,
                   alpha=0.095, max_depth=2,colsample_bytree=0.1, reg_lambda=1, min_child_weight=3)
xg_model=XGB.fit(X_Train.values, Y_Train)

pickle.dump(xg_model, open("model.pkl", "wb"))