import pandas as pd
import numpy as np

def load_and_prepare_student_data(file_path):
    df = pd.read_csv(file_path)
    df['G_avg'] = df[['G1', 'G2', 'G3']].mean(axis=1)

    # Combine Fedu and Medu into a single column named Medu
    df['Medu'] = df[['Fedu', 'Medu']].mean(axis=1)

    columns_to_keep = ['absences', 'failures', 'internet', 'higher', 'Medu', 'health', 'famsup', 'Pstatus', 'famrel', 'schoolsup', 'G_avg', 'paid', 'studytime']
    df_filtered = df[columns_to_keep]

    # Apply one-hot encoding to categorical variables
    df_encoded = pd.get_dummies(df_filtered, columns=['internet', 'higher', 'famsup', 'paid'], drop_first=True)

    # Ordinal encoding for ordinal categorical variables
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

    df_encoded.dropna(inplace=True)  # Drop rows with any NaN values
    df_encoded.reset_index(drop=True, inplace=True)  # Reset index after dropping rows

    # Ensure boolean columns are converted to integers
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)

    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()

    # Save the processed student dataset to a new CSV file
    df_encoded.to_csv("processed_student_data.csv", index=False)

    return df_encoded, labels, data

def load_and_prepare_adult_data(file_path):
    df = pd.read_csv(file_path)
    
    # Select relevant columns
    columns_to_keep = ['age', 'workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'hours.per.week', 'native.country', 'income']
    df_filtered = df[columns_to_keep]

    # Convert native.country to binary encoding (0 for United-States, 1 for all others)
    df_filtered['native.country'] = df_filtered['native.country'].apply(lambda x: 0 if x == 'United-States' else 1)

    # Convert categorical variables using one-hot encoding for other columns
    df_encoded = pd.get_dummies(df_filtered, columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex'], drop_first=True)

    # Binary encoding for income
    df_encoded['income'] = df_encoded['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # Drop any NaNs and reset index
    df_encoded.dropna(inplace=True)
    df_encoded.reset_index(drop=True, inplace=True)

    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()

    # Save the processed adult dataset to a new CSV file
    df_encoded.to_csv("processed_adult_data.csv", index=False)
    
    return df_encoded, labels, data

# Variable mapping for student dataset
student_variable_mapping = {
    'internet': 'internet_yes',
    'famsup': 'famsup_yes',
    'higher': 'higher_yes',
    'paid': 'paid_yes'
}

def apply_variable_mapping(variables, mapping):
    return [mapping.get(var, var) for var in variables]

# Example usage
# Replace 'student-por_raw.csv' and 'adult_cleaned.csv' with your actual file paths
student_data, student_labels, student_data_array = load_and_prepare_student_data('student-por_raw.csv')
adult_data, adult_labels, adult_data_array = load_and_prepare_adult_data('adult_cleaned.csv')
