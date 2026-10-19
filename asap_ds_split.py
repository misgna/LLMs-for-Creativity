import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 1. Load dataset
    df = pd.read_csv("dataset/asap_2_creative.csv")
    df["total"] = df['overall'].round()
    conditions = [df['total'] < 3, df['total'] == 3, df['total'] > 3]
    choices = [-1, 0, 1]
    
    # Corrected function name to np.select
    df['class'] = np.select(conditions, choices)
    
    # 2. Put the small prompt-engineering samples (125 rows total) into a temporary pool
    # The remaining majority of the data automatically becomes the 'test' set.
    prompt_pool, test = train_test_split(
        df, 
        train_size=125,  # 100 for train + 25 for dev
        random_state=42, 
        stratify=df['class']
    )
    
    # 3. Split the 125-row pool into your exact train (100) and dev (25) allocations
    train, dev = train_test_split(
        prompt_pool, 
        train_size=100, 
        test_size=25, 
        random_state=42, 
        stratify=prompt_pool['class']
    )
    
    # Save
    train.to_csv("dataset/train.csv", index=False)
    dev.to_csv("dataset/dev.csv", index=False)
    test.to_csv("dataset/test.csv", index=False)
    
    print("Dataset split is ready!")
    print(f"Train size (Prompt Engineering): {len(train)}")
    print(f"Dev size (Quick Evaluation):    {len(dev)}")
    print(f"Test size (Majority Evaluation): {len(test)}")
