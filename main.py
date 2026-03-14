from datasets import load_dataset

def get_data():
    # This pulls the data from HF servers to your local machine
    dataset = load_dataset("org_name/dataset_name")
    return dataset

if __name__ == "__main__":
    data = get_data()
    print("Dataset loaded successfully!")