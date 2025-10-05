# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from google.colab import userdata
from datasets import load_dataset, DatasetDict, Dataset

# Define constants for the dataset and output paths
hf_token = userdata.get('HF_TOKEN')
api = HfApi(token=hf_token)
DATASET_REPO_ID = "subrata2508/Tourism-Package-Prediction" # Replace with your actual dataset repo ID

# Load dataset from Hugging Face
dataset = load_dataset(DATASET_REPO_ID, token=hf_token)
df = dataset['train'].to_pandas() # Assuming the dataset has a 'train' split
print("Dataset loaded successfully.")

# Data Cleaning: Remove unnecessary columns
# Based on the data description, 'CustomerID' seems unnecessary for modeling.
# You might need to add more columns here if needed after exploring the data.
df = df.drop(['CustomerID'], axis=1)

# Handle potential missing values if any (replace with appropriate strategy)
# For example, you might fill missing numerical values with the mean or median,
# and missing categorical values with the mode or a placeholder.
# df.fillna(df.mean(), inplace=True) # Example for numerical columns
# df.fillna(df.mode().iloc[0], inplace=True) # Example for categorical columns

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify if the target variable is imbalanced
)

# Save the split datasets locally
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )

print("Data preparation complete. Train and test datasets saved locally and uploaded to Hugging Face.")
