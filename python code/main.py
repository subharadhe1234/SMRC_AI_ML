import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from traning import MultiRegressor
from pathlib import Path
from config import project_root
from dotenv import load_dotenv

load_dotenv()
current_dir= project_root
from helper import * 


field_name = os.getenv("field_name", "").split(",")
model_type_name = os.getenv("model_type_name", "").split(",") 


print(field_name)
for field in field_name:
    data_path = os.path.join(current_dir,  "dataset",f"{field}", f"df_{field}.csv")
    # read file
    print(data_path)
    df=pd.read_csv(data_path)
    df_predict=df.copy()
    # Splte X and y, X= Time  
    X=df.iloc[:,1:2].values #  X= Time 
    y=df.iloc[:,0:1].values #  y= field
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    for model_type in model_type_name:
        model_decition_tree=MultiRegressor(model_type=model_type)
        model_decition_tree.train(X_train,y_train)

        # test predict
        y_test_predict=model_decition_tree.predict(X_test)
        save_performance(model_name=f"{field} {model_type}",file_path="performance_matrix.csv",input_df=model_decition_tree.evaluate(y_test=y_test,mod="Test"),field= field)


        # train predict
        y_train_predict=model_decition_tree.predict(X_train)
        save_performance(model_name=f"{field} {model_type}",file_path="performance_matrix.csv",input_df=model_decition_tree.evaluate(y_test=y_train,mod="Train"),field= field)


        # data predict
        y_predict=model_decition_tree.predict(X)
        save_performance(model_name=f"{field} {model_type}",file_path="performance_matrix.csv",input_df=model_decition_tree.evaluate(y_test=y,mod="DataSet"),field= field)

        # =====================================
        df_predict[f"{model_type} predict"]=y_predict
        # create predict folder
        print(f"{model_type}  done ")
        # break

    save_path = os.path.join(current_dir, "output", field, "predict")
    os.makedirs(save_path, exist_ok=True)
    df_predict.to_csv(os.path.join(save_path,f"{field}_predict.csv"), index=False)


