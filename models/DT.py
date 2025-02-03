
from sklearn.tree import DecisionTreeRegressor
from graphs.error_graph import Plt
def DT_model(df_train_x, df_train_y, df_valid_x, df_valid_y, model_name,image_path):
    # Creating the Decision Tree model
    regressor = DecisionTreeRegressor(random_state=42, max_depth=9)

    # Training the model
    regressor.fit(df_train_x, df_train_y)

    # Plot the results and show metrics
    Plt(regressor, df_valid_x, df_valid_y, model_name,image_path)

