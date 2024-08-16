import os

directory = "C:\\Users\\adams\\OneDrive\\Desktop\\causal test\\causalvis_env\\lib\\site-packages\\causallearn"

# Function to replace np.mat with np.asmatrix
def replace_np_mat_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        content = content.replace("np.mat(", "np.asmatrix(")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Processed {file_path}")
    except UnicodeDecodeError:
        print(f"Skipping {file_path} due to encoding issues")

# Process each .py file
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            replace_np_mat_in_file(file_path)

print("np.mat has been replaced")