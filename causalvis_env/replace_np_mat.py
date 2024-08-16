import os

# Path to the causallearn library
directory = "C:\\Users\\adams\\OneDrive\\Desktop\\causal test\\causalvis_env\\lib\\site-packages\\causallearn"

# Function to replace np.mat with np.asmatrix
def replace_np_mat_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace("np.mat(", "np.asmatrix(")
    with open(file_path, 'w') as file:
        file.write(content)

# Walk through the directory and process each .py file
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            replace_np_mat_in_file(file_path)
            print(f"Processed {file_path}")

print("All occurrences of np.mat have been replaced with np.asmatrix.")