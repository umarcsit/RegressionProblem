import os
def output_dir(folder_name):

    output_file='/'+folder_name+'/'
    os.makedirs(output_file, exist_ok=True)
    results_img_path='./'+output_file

    dataset_path='./dataset'
    # Load the datasets
    path_train = dataset_path+'/training'
    path_valid = dataset_path+'/validation'

    training_files=os.listdir(path_train)
    validation_files=os.listdir(path_valid)
    return training_files,validation_files,results_img_path,path_train,path_valid