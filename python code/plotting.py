import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from traning import MultiRegressor
from helper import * 
from config import project_root
from dotenv import load_dotenv
# load env

current_dir= project_root
field_name = os.getenv("field_name", "").split(",")
model_type_name = os.getenv("model_type_name", "").split(",") # ,'svm'
# field=field_name[0]
load_dotenv()
for field in field_name:
    performance_path = os.path.join(current_dir, "output", "performance", "performance_matrix.csv")
    performance_matrix=pd.read_csv(performance_path)

    file_path = os.path.join(current_dir, "output", field, "predict",f"{field}_predict.csv")
    # read file
    df=pd.read_csv(file_path)

    X=df.iloc[:,1:2].values #  X= Time 
    y=df.iloc[:,0:1].values #  y= field
    for model_type in model_type_name:
        y_pred=df[f"{model_type} predict"]
        # get R2 and RMSE
        R2,RMSE = GET_R2_RMS(performance_matrix=performance_matrix,field=field,model_type=model_type)
        # ploting
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='red', label='Observed Data',facecolors='none', edgecolors='red')
        plt.scatter(X, y_pred,color='green',label=f'{model_type} : R2={R2:.4f}, RMSE={RMSE:.4f}')

        plt.xlabel('Time(S)')
        plt.ylabel('Soil Water Content ($cm^3$/$cm^3$)')
        plt.legend() # to show label
        plt.title(f'(SWRC)_{field} {model_type}')
        plt.tight_layout()

        # Save and show the plot
        save_img(plt_obj=plt,name=f'{field}_{model_type}',field=field)
        plt.close()
        # plt.show()

    final_image(field)


