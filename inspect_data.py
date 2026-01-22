import pandas as pd
import numpy as np

try:
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset (1).csv")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    
    if "Sleep Disorder" in df.columns:
        print("\nTarget Distribution:\n", df["Sleep Disorder"].value_counts(dropna=False))
    
    # Check for duplicates
    print("\nDuplicates:", df.duplicated().sum())
    
    # Check numerical stats
    print("\nNumerical Stats:\n", df.describe())
    
    # Check categorical stats
    print("\nCategorical Stats:\n", df.describe(include=['O']))

except Exception as e:
    print(f"Error: {e}")
