import pandas as pd
# Load datasets
def read_csv(train_path, valid_path):
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    train_x = train.iloc[:, 2:-1].fillna(train.iloc[:, 1:-1].mean())  # Fill missing values
    train_y = train.iloc[:, -1]
    valid_x = valid.iloc[:, 2:-1].fillna(valid.iloc[:, 1:-1].mean())
    valid_y = valid.iloc[:, -1]

    return train_x, train_y, valid_x, valid_y,train,valid