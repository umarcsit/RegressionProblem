from models.poly_regresion import poly_reg
from models.RNN_Model import RNNModel
from models.LSTM_Vanila import vanila_model
from models.LSTM_Stack import stack_model
from models.DT import DT_model
from data_augment.fwm_noise import add_fwm_noise
from data_augment.asw_noise import add_asw_noise
from data_augment.cm_noise import data_aug_cm_noised

def orignal_model(data,augment,img_path):
    org=[augment['dfx'],augment['dfy']]
    valid=[augment['dfvx'],augment['dfvy']]
    if data['poly_org']:
        poly_reg(org[0],org[1],valid[0],valid[1],'poly_org',img_path)
    if data['rnn_org']:
        RNNModel(org[0],org[1],valid[0],valid[1],'rnn_org',img_path)
    if data['vanila_org']:
       vanila_model(org[0],org[1],valid[0],valid[1],'vanila_org',img_path)
    if data['DT_org']:    
        DT_model(org[0],org[1],valid[0],valid[1], 'DT_org',img_path)
    if data['stack_org']:
        stack_model(org[0],org[1],valid[0],valid[1], 'stack_org',img_path)

def cm_model(data,augment,img_path):
    cm=[augment['cmx'],augment['cmy']]
    valid=[augment['dfvx'],augment['dfvy']]
    if data['stack_cm']:
        stack_model(cm[0],cm[1],valid[0],valid[1], 'stack_cm',img_path)
    if data['poly_cm']:     
        poly_reg(cm[0],cm[1],valid[0],valid[1],'poly_cm',img_path)
    if data['rnn_cm']:
        RNNModel(cm[0],cm[1],valid[0],valid[1],'rnn_cm',img_path)
    if data['vanila_cm']:    
        vanila_model(cm[0],cm[1],valid[0],valid[1], 'vanila_cm',img_path)
    if data['DT_cm']:
        DT_model(cm[0],cm[1],valid[0],valid[1], 'DT_cm',img_path)

def asw_model(data,augment,img_path):
    asw=[augment['aswx'],augment['aswy']]
    valid=[augment['dfvx'],augment['dfvy']]
    if data['rnn_asw']:
        RNNModel(asw[0],asw[1],valid[0],valid[1], 'rnn_asw',img_path)
    if data['poly_asw']:
        poly_reg(asw[0],asw[1],valid[0],valid[1],'poly_asw',img_path)
    if data['vanila_asw']:
        vanila_model(asw[0],asw[1],valid[0],valid[1],'vanila_asw',img_path)
    if data['stack_asw']:
        stack_model(asw[0],asw[1],valid[0],valid[1],'stack_asw',img_path)
    if data['DT_asw']:
        DT_model(asw[0],asw[1],valid[0],valid[1],'DT_asw',img_path)

def fwm_model(data,augment,img_path):
    fwm=[augment['fwmx'],augment['fwmy']]
    valid=[augment['dfvx'],augment['dfvy']]
    if data['poly_fwm']:
        poly_reg(fwm[0],fwm[1],valid[0],valid[1],'poly_fwm',img_path)
    if data['DT_fwm']:
        DT_model(fwm[0],fwm[1],valid[0],valid[1],'DT_fwm',img_path)
    if data['stack_fwm']:
        stack_model(fwm[0],fwm[1], valid[0],valid[1],'stack_fwm',img_path)
    if data['rnn_fwm']:
        RNNModel(fwm[0],fwm[1],valid[0],valid[1],'rnn_fwm',img_path)
    if data['vanila_fwm']:
        vanila_model(fwm[0],fwm[1],valid[0],valid[1],'vanila_fwm',img_path)

def model_run(data,aug_df,img_path):
    orignal_model(data,aug_df,img_path)
    asw_model(data,aug_df,img_path)
    fwm_model(data,aug_df,img_path)
    cm_model(data,aug_df,img_path)

def dataAugmented(train_x, train_y,valid_x, valid_y, target_size,data,img_path):
    cm_noised_x,cm_noised_y=data_aug_cm_noised(train_x,train_y)
    fwm_combined_x, fwm_combined_y = add_fwm_noise(train_x, train_y, target_size)
    asw_combined_x, asw_combined_y = add_asw_noise(train_x, train_y, target_size)
    aug_dic={'dfx':train_x,'dfy':train_y,'dfvx':valid_x,'dfvy':valid_y,'cmx':cm_noised_x,'cmy':cm_noised_y,'fwmx':fwm_combined_x,'fwmy':fwm_combined_y,'aswx':asw_combined_x,'aswy':asw_combined_y}
    model_run(data,aug_dic,img_path)