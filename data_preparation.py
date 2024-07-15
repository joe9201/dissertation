import pandas as pd

def load_and_prepare_student_data(file_path):
    df = pd.read_csv(file_path)
    df['G_avg'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    
    columns_to_keep = ['absences', 'failures', 'internet', 'higher', 'Medu', 'health', 'famsup', 'Pstatus', 'famrel', 'schoolsup', 'G_avg', 'paid', 'studytime']
    df_filtered = df[columns_to_keep]
    
    df_encoded = pd.get_dummies(df_filtered, columns=['internet', 'higher', 'famsup', 'paid'], drop_first=True)
    
    ordinal_map = {'Pstatus': {'A': 1, 'T': 2},
                   'famrel': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # Assuming 1-5 scale
                   'schoolsup': {'no': 0, 'yes': 1},
                   'health': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}  # Assuming 1-5 scale
    
    for col, mapping in ordinal_map.items():
        df_encoded[col] = df_encoded[col].map(mapping)
    
    df_encoded = df_encoded * 1
    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()
    
    return df_encoded, labels, data

def load_and_prepare_adult_data(file_path):
    df = pd.read_csv(file_path)
    # Add specific preparation steps for the adult dataset
    # This is an example; you will need to modify it based on your actual data
    
    columns_to_keep = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'income']
    df_filtered = df[columns_to_keep]
    
    df_encoded = pd.get_dummies(df_filtered, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'income'], drop_first=True)
    
    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()
    
    return df_encoded, labels, data

if __name__ == "__main__":
    student_file_path = 'data/student-por_raw.csv'
    adult_file_path = 'data/adult.csv'
    
    df_encoded_student, labels_student, data_student = load_and_prepare_student_data(student_file_path)
    print("Student data prepared and encoded.")
    
    df_encoded_adult, labels_adult, data_adult = load_and_prepare_adult_data(adult_file_path)
    print("Adult data prepared and encoded.")