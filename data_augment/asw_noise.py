import numpy as np
import pandas as pd
from sklearn.utils import shuffle
def add_asw_noise(train_x, train_y, target_size, noise_range=(0, 0.009), random_state=42):
    """
    Augments data by adding noise to one randomly selected feature for each augmented sample.

    Parameters:
        train_x (pd.DataFrame): Input features.
        train_y (pd.Series): Target values.
        target_size (int): Desired size of the augmented dataset.
        noise_range (tuple): Range of the noise to be added (default is (0, 0.009)).
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame, pd.Series: Augmented and shuffled feature set and target values.
    """
    if len(train_x) != len(train_y):
        raise ValueError("train_x and train_y must have the same number of samples.")
    if target_size <= len(train_x):
        raise ValueError("target_size must be greater than the current dataset size.")

    current_size = len(train_x)
    augmentation_factor = target_size - current_size
    np.random.seed(random_state)

    augmented_train_x = []
    augmented_train_y = []

    for _ in range(augmentation_factor):
        # Randomly select a row from train_x
        sampled_row = train_x.sample(n=1, random_state=random_state).values.flatten()
        # Randomly select one feature to modify
        random_feature_index = np.random.randint(0, train_x.shape[1])
        # Add noise to the selected feature
        noise = np.random.uniform(noise_range[0], noise_range[1])
        noisy_row = sampled_row.copy()
        noisy_row[random_feature_index] += noise
        # Append augmented row and corresponding target
        augmented_train_x.append(noisy_row)
        augmented_train_y.append(train_y.sample(n=1, random_state=random_state).values[0])

    # Convert augmented data to DataFrame
    augmented_train_x = pd.DataFrame(augmented_train_x, columns=train_x.columns)
    augmented_train_y = pd.Series(augmented_train_y)

    # Combine the original and augmented data
    combined_x = pd.concat([train_x, augmented_train_x], ignore_index=True)
    combined_y = pd.concat([train_y, augmented_train_y], ignore_index=True)

    # Shuffle combined dataset
    combined_x, combined_y = shuffle(combined_x, combined_y, random_state=random_state)

    return combined_x, combined_y