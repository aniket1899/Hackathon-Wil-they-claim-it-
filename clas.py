import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


def clas(df):

    '''
    This function applies classification algorithms from sklearn on train test
    split and return a dataframe with precision and recall value for each
    algorithm.
    '''
    x = df.drop("target_col", axis = 1)
    y = df.target_col
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, train_size = 0.7, random_state = 7)
    lg = LogisticRegression()
    rf = RandomForestClassifier(max_depth=4)
    dt = DecisionTreeClassifier(max_depth=4)
    gbc = GradientBoostingClassifier(max_depth=4)
    adbc = AdaBoostClassifier()
    xgb = XGBClassifier(max_depth= 4)
    di = {"LogisticRegression":[lg],"RandomForestClassifier":[rf],"DecisionTreeClassifier":[dt],\
            "GradientBoostingClassifier":[gbc],"AdaBoostClassifier":[adbc], "XGBoostClassifier":[xgb]}
    for k in di.keys():
        model = di[k][0]
        model.fit(X_train,Y_train)
        y_pred = model.predict(X_test)
        di[k].append(np.round(precision_score(Y_test, y_pred),2))
        di[k].append(np.round(recall_score(Y_test, y_pred),2))
        print(" ")
        print(k)
        print(" ")
        print(classification_report(Y_test, y_pred))
        print("-"*50)
    return pd.DataFrame({i:di[i][1:] for i in di.keys()}, index = ["precision","recall"])

