"""
Expand dataset100.csv to dataset300.csv by creating realistic variations.
Preserves statistical properties while generating synthetic samples.
"""

import pandas as pd
import numpy as np
from config.path import get_data_file

def expand_dataset(input_file, output_file, target_size=300):
    """
    Expand dataset while preserving statistical properties.
    Uses resampling with Gaussian noise to create realistic variations.
    """
    # Load original data
    df = pd.read_csv(get_data_file(input_file))
    original_size = len(df)

    print(f"Original dataset size: {original_size}")
    print(f"Target dataset size: {target_size}")
    print(f"Original columns: {len(df.columns)}")

    # Identify numeric columns (skip ID column)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'ＩＤ']

    # Calculate number of additional samples needed
    samples_needed = target_size - original_size

    print(f"Generating {samples_needed} synthetic samples...")

    # Create new samples
    new_samples = []
    np.random.seed(42)

    for i in range(samples_needed):
        # Randomly select a row to perturb
        sample_idx = np.random.randint(0, original_size)
        new_sample = df.iloc[sample_idx].to_dict()

        # Add small Gaussian noise to numeric columns
        for col in numeric_cols:
            if pd.notna(new_sample[col]):
                try:
                    col_std = df[col].std()
                    if col_std > 0:
                        noise_std = col_std * np.random.uniform(0.05, 0.10)
                        noise = np.random.normal(0, noise_std)

                        original_value = float(new_sample[col])
                        new_value = original_value + noise

                        col_min = df[col].min()
                        col_max = df[col].max()

                        # Clip to range
                        new_sample[col] = np.clip(new_value, col_min, col_max)
                except:
                    pass

        # Generate new ID
        new_sample['ＩＤ'] = f"expanded_{original_size + i + 1}"
        new_samples.append(new_sample)

    # Convert new samples to DataFrame
    new_df = pd.DataFrame(new_samples)

    # Concatenate with original data
    expanded_df = pd.concat([df, new_df], ignore_index=True, sort=False)

    # Shuffle rows
    expanded_df = expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save expanded dataset
    output_path = get_data_file(output_file)
    expanded_df.to_csv(output_path, index=False)

    print(f"\nExpanded dataset saved to: {output_path}")
    print(f"Final dataset size: {len(expanded_df)}")
    print(f"Final columns: {len(expanded_df.columns)}")
    print(f"Shape: {expanded_df.shape}")

    return expanded_df

if __name__ == '__main__':
    df = expand_dataset('dataset100.csv', 'dataset300.csv', target_size=300)
    print("\nDataset expansion complete!")
    print(f"First 3 rows:\n{df.head(3)}")
