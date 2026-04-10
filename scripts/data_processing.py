"""Data Extraction and Transformation (Cleaning) Process"""
import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    
    print("Data Extraction...")
    
    # Extract CSV Dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Extraction Complete: {df.shape[0]} rows found.")
    except FileNotFoundError:
        print(f"Error: File could not be found in {file_path}")
        return None

    # Transform the Dataset (Cleaning & Wrangling)
    print("Data Transformation...")

    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"{initial_rows - df.shape[0]} Duplicates were removed")

    # We fill 'person_emp_length' and 'loan_int_rate' with median values
    if 'person_emp_length' in df.columns:
        df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    if 'loan_int_rate' in df.columns:
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    #Outliers Handling (dropping abnormal length in years of experience, keeping as max the mode required for pension)
    if 'person_emp_length' in df.columns:
        df = df[df['person_emp_length'] < 50]
    
    #Outliers Handling (dropping abnormal length in years of age, keeping as max the regulatory ceiling for lending)
    if 'person_age' in df.columns:
         df = df[df['person_age'] < 75]


    print("Data Cleaning is done!")
    return df

if __name__ == "__main__":
    # Paths Definition for Operational Smoothness
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Connections for Raw and Processed Data
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'credit_risk_dataset.csv')
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_credit_risk.csv')

    cleaned_df = load_and_clean_data(raw_data_path)

    # LOAD the clean dataset as processed
    if cleaned_df is not None:
        cleaned_df.to_csv(processed_data_path, index=False)
        print(f"Successful Load: Cleaned dataset saved in {processed_data_path}")