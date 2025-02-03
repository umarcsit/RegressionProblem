import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def data_aug_cm_noised(train_x,train_y):
    # Data augmentation
    augmented_train_x = []
    augmented_train_y = []

    for _ in range(20):  # Repeat the data augmentation process 10 times
        noisy_train_x = train_x.copy()  # Start with the original data
        noise = np.random.normal(0, 0.09, train_x.shape) # Adjust the noise level (standard deviation)
        noisy_train_x = noisy_train_x + noise

        # Add the noisy data to the augmented datasets
        augmented_train_x.append(noisy_train_x)
        augmented_train_y.append(train_y)

    # Concatenate the augmented data with original data
    augmented_train_x = pd.concat(augmented_train_x)
    augmented_train_y = pd.concat(augmented_train_y)


    # Combine the original data with the augmented data
    combined_x = pd.concat([train_x, augmented_train_x], ignore_index=True)
    combined_y = pd.concat([train_y, augmented_train_y], ignore_index=True)


    # Shuffle the combined dataset
    combined_data = pd.concat([combined_x, combined_y], axis=1)
    combined_data = shuffle(combined_data)
    cm_combined_x = combined_data.iloc[:, :-1]
    cm_combined_y = combined_data.iloc[:, -1]
    return cm_combined_x,cm_combined_y