"""
This code is the application code for paper 
"Classifying raw irregular Landsat time series (CRIT) for large area land cover mapping by adapting Transformer model"
Three major processes:
    (1) load data and per-processing
    (2) manipulate data including creating training and testing data, make the data shape fits the model
    (3) model training and predcit.
"""
import os
import sys
import logging
import socket
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

import train_test_v2
import customized_train_v2
import transformer_encoder44_v2
from sklearn.model_selection import train_test_split
import importlib
print(socket.gethostname())
base_name = "this_log_"+socket.gethostname()

IS_TEST = 0 ## generate the model 
IS_TEST = 1 ## training and testing evaluation 

#*****************************************************************************************************************
## load csv file
csv_train_dir = '/mmfs1/scratch/jacks.local/khtran/Scripts/Transformer_LSP/LSP_samples/LSP_train_samples_v2_5.csv'
csv_test_dir = '/mmfs1/scratch/jacks.local/khtran/Scripts/Transformer_LSP/LSP_samples/LSP_test_samples_v2_5.csv'

data_train = pd.read_csv(csv_train_dir)
data_test = pd.read_csv(csv_test_dir)

version_train = np.asarray(data_train['version'])
version_test = np.asarray(data_test['version'])

indexs_train = np.asarray(data_train['indexs'])[version_train=='v1']
indexs_test = np.asarray(data_test['indexs'])[version_test=='v1']

years_train = np.asarray(data_train['years'])[version_train=='v1']
years_test = np.asarray(data_test['years'])[version_test=='v1']

pixel_info_train = np.asarray(data_train.iloc[:,1:10])[version_train=='v1']
pixel_info_test = np.asarray(data_test.iloc[:,1:10])[version_test=='v1']

labels_train_raw= np.asarray(data_train.iloc[:,10+12:986+12])[version_train=='v1']
labels_test_raw= np.asarray(data_test.iloc[:,10+12:986+12])[version_test=='v1']

features_train_raw= np.asarray(data_train.iloc[:,986+12:])[version_train=='v1']
features_test_raw= np.asarray(data_test.iloc[:,986+12:])[version_test=='v1']


## Feature Normalization
# Feature Normalization (arr-mean)/std
features_all = np.concatenate((features_train_raw,features_test_raw))
mean_features = np.nanmean(features_all)
std_features = np.nanstd(features_all) 
features_norm = (features_all-mean_features)/std_features # new_mean ~ 0 and new_std ~ 1
print("Feature shape:",features_all.shape) #N x 976
print("mean of features after normalization:",np.nanmean(features_norm))
print("Std of features after normalization:",np.nanstd(features_norm))

# Apply fill value of -9999 to nan values
features_norm[np.isnan(features_norm)]=-9999
print("Count fill value -9999 in features:",(features_norm==-9999).sum())

# Separate all features to train and test again
features_train=features_norm[:features_train_raw.shape[0],:]
features_test=features_norm[features_train_raw.shape[0]:,:]

## Label normalization
labels_all = np.vstack((labels_train_raw,labels_test_raw)) #N x 976
labels_all=labels_all.reshape(-1,244,4).reshape(-1,4) #N x 4
labels_min = np.tile(np.min(labels_all,axis=0), 244)
labels_max = np.tile(np.max(labels_all,axis=0), 244)
## Normalization
labels_train = (labels_train_raw - labels_min)/(labels_max - labels_min) 
labels_test = (labels_test_raw - labels_min)/(labels_max - labels_min)
print("Count fill value -9999 in labels:",(np.vstack((labels_train,labels_test))==-9999).sum())

list_years = np.unique(years_train)
print("List years:",list_years)

for yeari in list_years:
    features_train_yeari = features_train[years_train==yeari]
    features_test_yeari = features_test[years_test==yeari]
    print("Number of training samples in",yeari,":",features_train_yeari.shape[0])
    print("Number of testing samples in",yeari,":",features_test_yeari.shape[0])

# Expand dimensions to fit with transformer model if using only EVI2 time series (1 feature)
if len(features_train.shape)<3:
    features_train = np.expand_dims(features_train, axis=2)
    features_test = np.expand_dims(features_test, axis=2)
print(f"Features_train shape: {features_train.shape}")  # (N, 244, 1)
print(f"Labels_train shape: {labels_train.shape}")  # (N, 976)

print(f"Features_test shape: {features_test.shape}")  # (N, 244, 1)
print(f"Labels_test shape: {labels_test.shape}")  # (N, 976)
    
MODEL_DIR = "./model/"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

##*****************************************************************************************************************
## parameters 
minimal_N = 3 
KEPT_BLOCK_FT = 8; pre_train_version = "v2_1"
units = 64; head = 4; layern = 3
IMG_HEIGHT2 = 80   ; IMG_WIDTH2 = 7; IMG_BANDS2=1
BATCH_SIZE = 512;
YEARS = [2019, 2020]
LEARNING_RATE = 0.001; LAYER_N = layern; EPOCH = 5; ITERS = 1; L2 = 1e-3; METHOD=0; GPUi = 0
DROP = 0.1

MAX_L_IN16DAY = 4
PERIODS = 23 
DAYS = 16 
yr1 = 2000

#*****************************************************************************************************************


##*****************************************************************************************************************
## main function
if __name__ == "__main__":
    LEARNING_RATE = 0.001
    EPOCH = 200 
    EPOCH = 6
    METHOD = 2 # Hank
    ITERS = 1
    if IS_TEST==1:        ## 0 generate the model and 1 80/20 train/test
        # ITERS=5 
        ITERS=1            # try with one loop
    GPUi = 0
    print ("sys.argv n: " + str(len(sys.argv)))    
    ##***************************************************
    ## input parameters 
    DROP            = float(sys.argv[1])
    EPOCH           =   int(sys.argv[2] )
    METHOD          =   int(sys.argv[3] )
    LEARNING_RATE   = float(sys.argv[4])
    L2              = float(sys.argv[5])
    if len(sys.argv)>6:
        BATCH_SIZE       = int(sys.argv[6])    
    
    if len(sys.argv)>7:
        GPUi       = int(sys.argv[7])

    if len(sys.argv)>8:
        YEARS       = int(sys.argv[8])
        
    print ("BATCH_SIZE " + str(BATCH_SIZE))
    #*****************************************************************************************************************
    ## set GPU
    if '__file__' in globals():
        # base_name = os.path.basename(__file__)+socket.gethostname()
        base_name = os.path.basename(__file__)[17:]
        print(os.path.basename(__file__))
    
    yr1=YEARS
    if isinstance(YEARS,list):
        yr1="all"
    
    base_name = base_name+'.year'+str(yr1)+'.layer'+str(LAYER_N)+'.dim'+str(IMG_HEIGHT2)+'.METHOD'+str(METHOD)+'.DROP'+str(DROP)+'.B'+str(BATCH_SIZE)+'.LR'+str(LEARNING_RATE)+'.L2'+str(L2)
    #print (base_name[9:15])

    if METHOD==0:
        logging.basicConfig(filename=base_name+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    print (GPUi)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')        
    print(tf.config.get_visible_devices())
    logging.info (tf.config.get_visible_devices()[0])
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] 
    #*****************************************************************************************************************
    ## get train and testing data   
    importlib.reload(train_test_v2)
    IMG_WIDTH2 = 8
    XY_DIM_N = 4
    proportion=1.0
    if IS_TEST==1:
        proportion=0.8

    IMG_HEIGHT2=80
    IMG_WIDTH2 = 8
    
    trainx_transformer = features_train.copy()# K changed on Sep 19 2023
    train_n = trainx_transformer.shape[0]   
    per_epoch = train_n//BATCH_SIZE
    print ("Train n = " + str(train_n) )
    #*****************************************************************************************************************
    print ("\n\n#partical CNN and transformers*************************************************************************\n\n\n")
    accuracylist1 = list()
    #*****************************************************************************************************************
    ## model 1  13 * 3
    accuracylist2 = list()        
    #*****************************************************************************************************************
    ## model any length transformer 
    units = 64; head = 4; layern = 3
    per_epoch = train_n//BATCH_SIZE
    validation_split = 0
    if IS_TEST==1:
        validation_split=0.04
        
    strategy = tf.distribute.MirroredStrategy()
    # exit()
    for i in range(ITERS):
        print_str = "\n {:3d}: transformer model ********************************************************************************iter".format(i+1)
        print (print_str); logging.info (print_str)

        ## ****************************************************************
        ## prediction model construction (with N data)
        model_drop = DROP 
        importlib.reload(transformer_encoder44_v2)
        model = transformer_encoder44_v2.get_transformer_new_att0_daily_withsensor(n_times=features_train.shape[1],n_feature=features_train.shape[2],n_out=4,
                                                                                layern=layern, units=units, n_head=head, drop=model_drop,
                                                                                is_day_input=False,is_sensor=False, is_sensor_embed=False, is_xy=False, xy_n=XY_DIM_N,is_reflectance=True)  # dense_layer_n=PRE_TRAIN
                                                                                
        if i==0:
            print (model.summary())        
        
        ## **************************************************************************************************************************************
        ## fine-tuning the model without transfer learning Or transfer learned model if PRE_TRAIN is True
        importlib.reload(customized_train_v2)          
        trainx_transformer = features_train.copy()# K changed on Sep 19 2023
        trainy_transformer = labels_train.copy() # K changed on Sep 19 2023
        trainy_transformer2 = trainy_transformer.reshape (trainy_transformer.shape[0],244,4)
        #trainy_transformer_copy = trainy_transformer.copy()
        #trainy_transformer = trainy_transformer/100
        print("Loss function is MSE")
        model_history = customized_train_v2.my_train_1schedule(model,trainx_transformer,trainy_transformer2,epochs=EPOCH,start_rate=LEARNING_RATE,\
            loss='mse',per_epoch=per_epoch,split_epoch=5,option=METHOD,decay=L2,batch_size=BATCH_SIZE,validation_split=validation_split)
                
        #for yeari in YEARS_LIST:
        #    print (yeari)
        testx_transformer = features_test.copy()# K changed on Sep 19 2023
        testy_transformer = labels_test.copy() # K changed on Sep 19 2023
        testy_transformer2 = testy_transformer.reshape(testy_transformer.shape[0],244,4)
        r,p_value,mad,rmse,msb,labels_testout,labels_testout_std,labels_pred,labels_pred_std,labels_pred_masked,labels_pred_std_masked,labels_pred2 = customized_train_v2.test_lsp_accuacy(model,testx_transformer,testy_transformer2,labels_min.reshape(-1,244,4),labels_max.reshape(-1,244,4))
        print (">>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics Pearson_R:" , np.round(r,2))
        print (">>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics p_value:  " , np.round(p_value,4))
        print (">>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics MAD:      " , np.round(mad,1))
        print (">>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics RMSE:     " , np.round(rmse,1))
        print (">>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics MSB:      " , np.round(msb,1))
        
        df_out = pd.DataFrame()
        pixel_info = ["version","index","years","NLCD","row","col","SiteID","HLStile","subset"]
        for k in range(len(pixel_info)): #9 information
            colname = pixel_info[k]
            df_out[colname]=pixel_info_test[:,k]
        onsets = ["G","M","S","D"]
        onsets_std = ["G_std","M_std","S_std","D_std"]
        for k in range(len(onsets)): #four onsets
            colname = onsets[k]+ "_test"
            df_out[colname]=labels_testout[:,k]
        for k in range(len(onsets_std)): #four onsets std
            colname = onsets_std[k]+ "_test"
            df_out[colname]=labels_testout_std[:,k]    
        for k in range(len(onsets)): #four predicted onsets
            colname = onsets[k]+ "_pred"
            df_out[colname]=labels_pred[:,k]
        for k in range(len(onsets_std)): #four predicted onsets std
            colname = onsets_std[k]+ "_pred"
            df_out[colname]=labels_pred_std[:,k] 
        for k in range(len(onsets)): #four predicted onsets
            colname = onsets[k]+ "_pred_masked"
            df_out[colname]=labels_pred_masked[:,k]
        for k in range(len(onsets_std)): #four predicted onsets std
            colname = onsets_std[k]+ "_pred_masked"
            df_out[colname]=labels_pred_std_masked[:,k] 
            
        for k in range(976): #976 predictions
            if k%4==0:
                xcolname = 'G_' + str(int(k/4)+1)+'_pred'
            elif k%4==1:
                xcolname = 'M_' + str(int(k/4)+1)+'_pred'
            elif k%4==2:
                xcolname = 'S_' + str(int(k/4)+1)+'_pred'
            elif k%4==3:
                xcolname = 'D_' + str(int(k/4)+1)+'_pred'
            df_out[xcolname]=labels_pred2[:,k] 
            
        colnames=[]   
        for k in range(features_test_raw.shape[1]): #four onsets
            colname = 'EVI2_doy'+str(k*3+1) # 3-day composite
            colnames.append(colname)
        df_features_test = pd.DataFrame(features_test_raw,columns=colnames) #reshape to 2D array.
        df_out=pd.concat((df_out,df_features_test),axis=1)
        


        accuracylist2.append (rmse)
        
        model_name = MODEL_DIR+base_name+'.model.ITER'+str(i)+'.h5'
    
        model.save(model_name)
        csv_name = MODEL_DIR+base_name+'.predict.ITER'+str(i)+'.csv'
        df_out.to_csv(csv_name)
    #*****************************************************************************************************************
    ## model  LSTM 
    accuracylist3 = list()
    
    #*****************************************************************************************************************
    ## print accuacy 
    print (accuracylist1)
    print (accuracylist2)
    # print (accuracylist3)
    # i=0
    # for yeari in YEARS_LIST: 
        # acc_index = np.array(range(i,len(accuracylist1),len(YEARS_LIST)))
        # if accuracylist1!=[] and acc_index.size>0:
            # print ('{:4d}'.format(yeari)+" year accuracylist1 rf mean" + '  {:4.2f}'.format(np.array(accuracylist1)[acc_index].mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist1)[acc_index].std()*100) )
        
        # acc_index = np.array(range(i,len(accuracylist2),len(YEARS_LIST)))
        # if accuracylist2!=[] and acc_index.size>0:
            # print ('{:4d}'.format(yeari)+" year accuracylist2 2d mean" + '  {:4.2f}'.format(np.array(accuracylist2)[acc_index].mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist2)[acc_index].std()*100) )
        
        # acc_index = np.array(range(i,len(accuracylist3),len(YEARS_LIST)))
        # if accuracylist3!=[] and acc_index.size>0:
            # print ('{:4d}'.format(yeari)+" year accuracylist3 1d mean" + '  {:4.2f}'.format(np.array(accuracylist3)[acc_index].mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist3)[acc_index].std()*100) )
        
        # i=i+1
##################################################################################################################################################


 
