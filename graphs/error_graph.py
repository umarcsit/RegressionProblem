import matplotlib.pyplot as plt
from global_list import file_image,R2_list,MAE_list,MSE_list,MAPE_list,RMSE_list,Huber_list,model_list,pred_list
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf  # Importing TensorFlow for Huber loss
def errors_stored(model_name,mse,mae,r2,rmse,mape,huber_loss):
    model_list.append(model_name)
    MSE_list.append(mse)
    MAE_list.append(mae)
    R2_list.append(r2)
    RMSE_list.append(rmse)
    MAPE_list.append(mape)
    Huber_list.append(huber_loss)
def fig_plot(valid_pred, valid_y, model_name, metrics,image_path):  
    graphplot(model_name,metrics['MSE'],metrics['MAE'],metrics['RMSE'],metrics['R2'],metrics['MAPE'],metrics['Huber'],valid_y,valid_pred,image_path)
    errors_stored(model_name,metrics['MSE'],metrics['MAE'],metrics['R2'],metrics['RMSE'],metrics['MAPE'],metrics['Huber'])

def graphplot(model_name,mse,mae,rmse,r2,mape,huber_loss,y_val,y_pred,image_path):
        # Prepare the title with error metrics
    title = f'{model_name} Model: Actual vs Predicted Values on Validation Data\n' \
            f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f},\n R2: {r2:.4f}, MAPE: {mape:.4f}, Huber Loss: {huber_loss:.4f}'

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_val, label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    # plt.ylim([0, 1.3])
    plt.legend()
    plt.grid(True)
    file_image.append(model_name + '.jpg')
    plt.savefig(image_path+ model_name + '.jpg')
    plt.show()
def Plt(model, valid_x, valid_y, model_name,image_path):
    # Predict on the validation data
    y_pred = model.predict(valid_x)
    y_pred = np.array(y_pred).flatten()
    pred_list.append(np.round(y_pred,3))
    # Calculating the error metrics
    mse = mean_squared_error(valid_y, y_pred)
    mae = mean_absolute_error(valid_y, y_pred)
    r2 = r2_score(valid_y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(valid_y, y_pred)

    # Calculating Huber loss on validation data
    huber_loss_fn = tf.keras.losses.Huber()
    huber_loss = huber_loss_fn(valid_y, y_pred).numpy()
    # Reshape data for comparison (if necessary)
    y_val = np.array(valid_y).flatten()
    y_pred = np.array(y_pred).flatten()
    graphplot(model_name,mse,mae,rmse,r2,mape,huber_loss,y_val,y_pred,image_path)
    errors_stored(model_name,mse,mae,r2,rmse,mape,huber_loss)