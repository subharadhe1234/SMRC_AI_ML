import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from config import project_root


current_dir= project_root
load_dotenv()


field_name = os.getenv("field_name", "").split(",")
data_path = os.path.join(current_dir,"dataset", "feeds.csv")
# read file
df=pd.read_csv(data_path)

# drop the useless features
df=df.drop(['latitude','longitude','elevation','status','entry_id'],axis=1)

# Convert the 'created_at' column to seconds since the epoch
df['created_at_seconds'] = df['created_at'].apply(
    lambda x: int(datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp())
)
base_time=df['created_at_seconds'].iloc[0]
df["relative_time"] = df["created_at_seconds"] - base_time
df.drop(['created_at','created_at_seconds'],axis=1,inplace=True)

for field in field_name:
    # print(field)
    df_field=pd.DataFrame(columns=[["field","time"]])
    df_field[["field","time"]]=df[[f"{field}","relative_time"]]
    save_path = os.path.join(current_dir, "dataset",f"{field}")
    os.makedirs(save_path, exist_ok=True)
    df_field.to_csv(os.path.join(save_path,f"df_{field}.csv"), index=False)




