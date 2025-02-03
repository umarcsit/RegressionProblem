import numpy as np
import pandas as pd
from sklearn.utils import shuffle
# Data augmentation
def add_fwm_noise(train_x, train_y, target_size, random_state=42):
    current_size = len(train_x)
    augmentation_factor = target_size - current_size
    np.random.seed(random_state)

    augmented_train_x = []
    augmented_train_y = []

    for _ in range(augmentation_factor):
        noisy_row = train_x.sample(n=1).values + np.random.uniform(0, 0.009, train_x.shape[1])
        augmented_train_x.append(noisy_row.flatten())
        augmented_train_y.append(train_y.sample(n=1).values[0])

    # Convert augmented data to DataFrame
    augmented_train_x = pd.DataFrame(augmented_train_x, columns=train_x.columns)
    augmented_train_y = pd.Series(augmented_train_y)

    # Combine the original and augmented data
    combined_x = pd.concat([train_x, augmented_train_x], ignore_index=True)
    combined_y = pd.concat([train_y, augmented_train_y], ignore_index=True)

    # Shuffle combined dataset
    combined_data = pd.concat([combined_x, combined_y], axis=1)
    combined_data = shuffle(combined_data, random_state=random_state)
    combined_x = combined_data.iloc[:, :-1]
    combined_y = combined_data.iloc[:, -1]

    return combined_x, combined_y
