import pandas as pd
from datasets import load_dataset

def fetch_and_clean_data(repo_id: str) -> pd.DataFrame:
    # 1. Ingest Data: Map from Hugging Face to local memory
    print(f"Fetching dataset from {repo_id}...")
    dataset = load_dataset(repo_id)
    df = dataset['train'].to_pandas()
    
    print(f"Initial matrix dimensions: {df.shape}")

    # 2. Eliminate Degeneracy: Remove identical row vectors
    df.drop_duplicates(inplace=True)

    # 3. Resolve Undefined Elements (NaN)
    # Isolate the numerical and categorical subspaces
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Impute numerical features using the expected value (mean) of the column vector
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Impute categorical features using the most frequent element (mode)
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Domain Enforcement: Drop any remaining vectors that failed imputation
    df.dropna(inplace=True)

    print(f"Cleaned matrix dimensions: {df.shape}")
    return df

if __name__ == "__main__":
    # Execute the pipeline
    cleaned_dataframe = fetch_and_clean_data("Amaan/DataDays")
    
    # Output the topology of the resulting space
    print("\n--- Final Data Structure ---")
    print(cleaned_dataframe.info())