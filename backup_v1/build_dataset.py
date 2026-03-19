"""
Entry point wrapper script to build the dataset.
"""
from config import RAW_DATA_DIR, DATASET_FILENAME
from src.dataset_builder import build_dataset

def main():
    print("Starting dataset building pipeline...")
    
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' does not exist.")
        return
        
    df = build_dataset(RAW_DATA_DIR)
    
    if df is None or df.empty:
        print("Dataset is empty. Please check your data folders and files.")
        return
        
    # Save the dataset
    df.to_csv(DATASET_FILENAME, index=False)
    print(f"Dataset successfully saved to {DATASET_FILENAME}")
    print(f"Total records in dataset: {len(df)}")
    
    if 'bankruptcy_label' in df.columns:
        bankrupt_count = df['bankruptcy_label'].sum()
        print(f"Bankrupt instances labeled '1': {bankrupt_count}")
        print(f"Non-bankrupt instances labeled '0': {len(df) - bankrupt_count}")

if __name__ == "__main__":
    main()
