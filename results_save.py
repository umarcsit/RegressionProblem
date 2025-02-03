from global_list import R2_list,MAE_list,MSE_list,MAPE_list,pred_list,RMSE_list,Huber_list,model_list,file_image
import pandas as pd

def csv_save(data,valid,valid_y):
    org_results=pd.DataFrame({'Model':model_list,'MSE':MSE_list,'MAE':MAE_list,'R2':R2_list,'RMSE':RMSE_list,'MAPE':MAPE_list,'Huber':Huber_list,'FileName':file_image})
    org_results.sort_values(by='MSE',ascending=True).to_csv(data['error_dir']+'.csv',index=False)
    for n in range (len(pred_list)):
        pred_list[n]=pred_list[n].tolist()
    df_dict={}
    df_dict['vdo_id']=valid['uploaded_video_id']
    df_dict['ActualValue']=valid_y.tolist()
    for n in range(len( model_list)):
        df_dict[model_list[n]]=pred_list[n]
    # vdo_id=pd.concat([train['uploaded_video_id'],valid['uploaded_video_id']],axis=0,ignore_index=True)
    pd.DataFrame(df_dict).to_csv( data['data_point']+'.csv',index=False)