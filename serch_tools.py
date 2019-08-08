# manipulacion y analisis de datos
import pandas as pd
import numpy as np
#visualizacion de datos
import matplotlib.pyplot as plt
import seaborn as sns

#ML preprocessing
from sklearn.model_selection import train_test_split

#ML Evaluation
from sklearn import metrics

#ML Interpretation
import shap

def check_inicio_final(df,n=5):
    return pd.concat([df.head(n),df.tail(n)])


def train_test_valid_split(df, x_cols, y_col):
    
    x_train_test, x_valid, y_train_test, y_valid = train_test_split(
    df[x_cols],
    df[y_col],
    test_size=0.1,
    random_state=8)

    x_train, x_test, y_train, y_test = train_test_split(x_train_test, 
                                                    y_train_test,
                                                    test_size = 0.2,
                                                    random_state=10)
    
    return x_train, x_test, x_valid, y_train, y_test, y_valid

def report_evaluation_regression(y_real, y_pred):
    
    print("R*2: %1.4f" % metrics.r2_score(y_real,y_pred))
    print("MEA: %1.4f" % metrics.mean_absolute_error(y_real,y_pred))
    print("MSE %1.3f" % metrics.mean_squared_error(y_real,y_pred))
    print("RMSE %1.3f" % np.sqrt(metrics.mean_squared_error(y_real,y_pred)))
    
def resi_plot(y_real, y_pred):
    
    res = (y_real - y_pred) / y_real
    plt.scatter(y_real, res, alpha=0.5)
    print("residuo_porcentual: %1.4f" % res.mean())
    
def yy_plot(y_real, y_pred):
    plt.scatter(y_real, y_pred, alpha=0.5)
    mi2,ma2 = y_real.min(), y_real.max()
    mi1,ma1 = y_pred.min(), y_pred.max()
    plt.plot([mi1,ma1],[mi2,ma2], linestyle='--', color='navy')
    plt.ylabel('Y_real')
    plt.xlabel('Y_estimada')
    
    
def shap_shap(clf, x_valid):
    explainer = shap.TreeExplainer(clf)
    return explainer, explainer.shap_values(x_valid)
    
    
def shap_summary(shap_values, x_valid):
    
    shap.initjs()
    shap.summary_plot(shap_values, x_valid)
    
def shap_lime(explainer, shap_values, X,i):
    
    shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:])