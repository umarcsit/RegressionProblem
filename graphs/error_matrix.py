def calculate_metrics(valid_pred, valid_y):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    import numpy as np
    # Mean Squared Error (MSE)
    mse = mean_squared_error(valid_y, valid_pred)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(valid_y, valid_pred)
    # R-squared (R²)
    r2 = r2_score(valid_y, valid_pred)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(valid_y, valid_pred) * 100
    # Huber Loss (calculated manually)
    delta = 1.0  # You can adjust delta based on your needs
    residual = np.abs(valid_y - valid_pred)
    huber = np.where(residual <= delta, 0.5 * residual**2, delta * (residual - 0.5 * delta)).mean()
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape,
        'Huber': huber
    }