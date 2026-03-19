"""
Entry point wrapper script to build the dataset.

Usage:
    python build_dataset.py
"""
from config import RAW_DATA_DIR, DATASET_FILENAME, CANDIDATE_FEATURES, TARGET
from src.dataset_builder import build_dataset
from src.preprocessing import preprocess_data


def main():
    print("=" * 60)
    print("  FINANCIAL DISTRESS PREDICTION — DATASET BUILDER")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' does not exist.")
        return

    # ── Build raw dataset ─────────────────────────────────────────────────
    df = build_dataset(RAW_DATA_DIR)

    if df is None or df.empty:
        print("Dataset is empty. Please check your data folders and files.")
        return

    # ── Preprocess ────────────────────────────────────────────────────────
    print("\nPreprocessing dataset...")
    df = preprocess_data(df)

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(DATASET_FILENAME, index=False)
    print(f"\n{'=' * 60}")
    print(f"Dataset saved to: {DATASET_FILENAME}")
    print(f"Total records: {len(df)}")

    if TARGET in df.columns:
        distress_count = int(df[TARGET].sum())
        healthy_count = len(df) - distress_count
        print(f"  Distressed (label=1): {distress_count}")
        print(f"  Healthy    (label=0): {healthy_count}")
        print(f"  Distress ratio: {distress_count / len(df):.2%}")

    # Feature availability summary
    features_present = [f for f in CANDIDATE_FEATURES if f in df.columns]
    print(f"\nFeature count: {len(features_present)} / {len(CANDIDATE_FEATURES)} candidate features available")
    print(f"  Financial ratios:   {sum(1 for f in features_present if not any(s in f for s in ['_lag', '_x_', 'gdp','inflation','interest','usdtry','unemployment','industrial','credit','m2']))}")
    print(f"  Macro (current):    {sum(1 for f in features_present if f in __import__('config').MACRO_FEATURES)}")
    print(f"  Macro (lagged):     {sum(1 for f in features_present if '_lag' in f)}")
    print(f"  Interaction terms:  {sum(1 for f in features_present if '_x_' in f)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
