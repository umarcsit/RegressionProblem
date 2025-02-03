from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from graphs.error_matrix import calculate_metrics
from graphs.error_graph import fig_plot
from global_list import pred_list
import numpy as np
def poly_reg(train_x, train_y, valid_x, valid_y, data_nature,image_path):
    poly = PolynomialFeatures(degree=2)  # You can adjust the degree
    X_poly = poly.fit_transform(train_x)
    X_poly_valid = poly.transform(valid_x)

    # Fit Ridge regression model with alpha=100
    ridge_reg = Ridge(alpha=100)
    ridge_reg.fit(X_poly, train_y)

    # Make predictions
    valid_pred = ridge_reg.predict(X_poly_valid)
    pred_list.append(np.round(valid_pred,2))
    # Calculate metrics
    metrics = calculate_metrics(valid_pred, valid_y)
    # Plot results
    fig_plot(valid_pred, valid_y, data_nature + '_Poly_Regression', metrics,image_path)