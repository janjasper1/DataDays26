from datasets import load_dataset

# This specifically connects to the Amaan/DataDays dataset
dataset = load_dataset("Amaan/DataDays")

# Preview the data to ensure connection is live
print(dataset)

# Example: access the 'train' split
if 'train' in dataset:
    print(dataset['train'][0])