import kagglehub

# Download latest version
path = kagglehub.dataset_download("ziya07/marketing-behavior-prediction-dataset")

print("Path to dataset files:", path)