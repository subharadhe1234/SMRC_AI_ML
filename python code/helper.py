import openpyxl
import os
import matplotlib.pyplot as plt
from pathlib import Path
from config import project_root
from dotenv import load_dotenv
# load env
load_dotenv()
current_dir= project_root

import os
import pandas as pd

def save_performance(model_name, file_path, input_df, field):
    """
    Update test performance metrics for a model in a CSV file based on input DataFrame.

    :param model_name: Name of the model to save/update
    :param file_path: Name of the CSV file to save the performance
    :param input_df: DataFrame with two columns: 'Metric' and 'Value' to be updated for the model
    :param field: Field name for dynamic path creation
    """

    save_path = os.path.join(current_dir, "output", "performance", f"{file_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define column names
    headers = ["Model", "Test R2", "Test RMSE", "Train R2", "Train RMSE", "DataSet R2", "DataSet RMSE"]

    try:
        # Load the existing CSV file if it exists
        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
        else:
            # Create a new DataFrame if the file does not exist
            df_existing = pd.DataFrame(columns=headers)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check if the model already exists in the DataFrame
    if model_name in df_existing["Model"].values:
        # Update the existing row for the model
        for _, row in input_df.iterrows():
            metric = row["Metric"]
            value = row["Value"]
            if metric in df_existing.columns:
                df_existing.loc[df_existing["Model"] == model_name, metric] = value
    else:
        # Add a new row for the model
        new_row = {col: None for col in headers}  # Initialize a dictionary for the new row
        new_row["Model"] = model_name
        for _, row in input_df.iterrows():
            metric = row["Metric"]
            value = row["Value"]
            if metric in headers:
                new_row[metric] = value
        
        # Create a DataFrame from the new row
        new_row_df = pd.DataFrame([new_row])
        # Exclude all-NA entries before concatenation
        new_row_df = new_row_df.dropna(how="all", axis=1)
        df_existing = pd.concat([df_existing, new_row_df], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    df_existing.to_csv(save_path, index=False)



# save image
def save_img(plt_obj, name,field):
    """
    Function to save a given plot object to the specified path.
    :param plt_obj: Matplotlib plot object.
    :param save_path: Path to save the plot.
    """
    save_path = os.path.join(current_dir, "output", field, "image")
    os.makedirs(save_path, exist_ok=True)
    path=f"{save_path}/{name}.png"
    plt_obj.savefig(path)
    print(f"Plot saved at {path}")

def GET_R2_RMS(performance_matrix,field,model_type):
    for index, row in performance_matrix.iterrows():
                if row["Model"]==f"{field} {model_type}":
                    R2=row["DataSet R2"]
                    RMSE=row["DataSet RMSE"]
                    return R2,RMSE

def final_image(field):
    model_type_name = os.getenv("model_type_name", "").split(",") 
    performance_path = os.path.join(current_dir,  "output", "performance", "performance_matrix.csv")
    performance_matrix=pd.read_csv(performance_path)

    file_path = os.path.join(current_dir,  "output", field, "predict",f"{field}_predict.csv")
    # read file
    df=pd.read_csv(file_path)

    X=df.iloc[:,1:2].values #  X= Time 
    y=df.iloc[:,0:1].values #  y= field
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Observed Data',facecolors='none', edgecolors='red')
    for model_type in model_type_name:
        y_pred=df[f"{model_type} predict"]
        R2,RMSE=GET_R2_RMS(model_type=model_type,performance_matrix=performance_matrix,field=field)
        plt.scatter(X, y_pred,label=f'{model_type} : R2={R2:.4f}, RMSE={RMSE:.4f}')

    plt.xlabel('Time(S)')
    plt.ylabel('Soil Water Content ($cm^3$/$cm^3$)')
    plt.legend() # to show label
    plt.title(f'(SWRC)_final_image {model_type}')
    plt.tight_layout()

    # Save and show the plot
    save_img(plt_obj=plt,name=f'{field}_final_image',field=field)
    plt.close()
    # plt.show()