import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_student_data(file_path):
    # calculate the average grade
    df = pd.read_csv(file_path)
    df['G_avg'] = df[['G1', 'G2', 'G3']].mean(axis=1)

    # Combine parental education levels into a single column
    df['Medu'] = df[['Fedu', 'Medu']].mean(axis=1)

    columns_to_keep = [
        'absences', 'failures', 'internet', 'higher', 'Medu', 'health', 
        'famsup', 'Pstatus', 'famrel', 'schoolsup', 'G_avg', 'paid', 'studytime'
    ]
    df_filtered = df[columns_to_keep]

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_filtered, columns=['internet', 'higher', 'famsup', 'paid'], drop_first=True)

    # Map ordinal categories to numeric values
    ordinal_map = {
        'Pstatus': {'A': 1, 'T': 2},
        'famrel': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        'schoolsup': {'no': 0, 'yes': 1},
        'health': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        'Medu': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    }

    for col, mapping in ordinal_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

    df_encoded.dropna(inplace=True)
    df_encoded.reset_index(drop=True, inplace=True)

    # boolean columns to integers
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)

    # Prepare the labels and data
    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()

    # Save the dataset
    df_encoded.to_csv("processed_student_data.csv", index=False)

    return df_encoded, labels, data

def encode_categorical_features(df, columns):
    for col in columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def load_and_prepare_adult_data(file_path):
    df = pd.read_csv(file_path)
    columns_to_keep = [
        'age', 'workclass', 'education', 'marital.status', 
        'occupation', 'relationship', 'race', 'sex', 'hours.per.week', 
        'native.country', 'income'
    ]
    df_filtered = df[columns_to_keep].copy()

    # Binary encoding for native country
    df_filtered['native.country'] = df_filtered['native.country'].apply(lambda x: 0 if x == 'United-States' else 1)

    # Encode all categorical features
    categorical_columns = [
        'workclass', 'education', 'marital.status', 
        'occupation', 'relationship', 'race', 'sex'
    ]
    df_encoded = encode_categorical_features(df_filtered, categorical_columns)

    # Binary encoding for income
    df_encoded['income'] = df_encoded['income'].apply(lambda x: 1 if x == '>50K' else 0)

    df_encoded.dropna(inplace=True)
    df_encoded.reset_index(drop=True, inplace=True)

    # Prepare the labels and data
    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()

    # Save dataset
    df_encoded.to_csv("processed_adult_data.csv", index=False)
    
    return df_encoded, labels, data

def create_smaller_student_dataset(file_path):
    df_encoded, _, _ = load_and_prepare_student_data(file_path)
    columns_to_keep = ['absences', 'G_avg', 'Medu', 'failures', 'studytime', 'higher_yes', 'health']
    df_smaller_student = df_encoded[columns_to_keep].copy()
    df_smaller_student.to_csv("smaller_student_dataset.csv", index=False)
    return df_smaller_student

def create_smaller_adult_dataset(file_path):
    df_encoded, _, _ = load_and_prepare_adult_data(file_path)
    columns_to_keep = ['income', 'hours.per.week', 'age', 'education', 'workclass', 'occupation', 'native.country']
    df_smaller_adult = df_encoded[columns_to_keep].copy()
    df_smaller_adult.to_csv("smaller_adult_dataset.csv", index=False)
    return df_smaller_adult

# Variable mapping for student
student_variable_mapping = {
    'internet': 'internet_yes',
    'famsup': 'famsup_yes',
    'higher': 'higher_yes',
    'paid': 'paid_yes'
}

def apply_variable_mapping(variables, mapping):
    return [mapping.get(var, var) for var in variables]

student_data, student_labels, student_data_array = load_and_prepare_student_data(r'C:\Users\adams\OneDrive\Desktop\causal test\data\student-por_raw.csv')
adult_data, adult_labels, adult_data_array = load_and_prepare_adult_data(r'C:\Users\adams\OneDrive\Desktop\causal test\data\adult_cleaned.csv')

# Create smaller datasets
smaller_student_data = create_smaller_student_dataset(r'C:\Users\adams\OneDrive\Desktop\causal test\data\student-por_raw.csv')
smaller_adult_data = create_smaller_adult_dataset(r'C:\Users\adams\OneDrive\Desktop\causal test\data\adult_cleaned.csv')