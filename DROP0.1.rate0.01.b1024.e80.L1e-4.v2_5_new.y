gpu003
Feature shape: (132364, 244)
mean of features after normalization: 4.664527319692421e-17
Std of features after normalization: 1.0000000000000018
Count fill value -9999 in features: 22909085
Count fill value -9999 in labels: 0
List years: [2019 2020]
Number of training samples in 2019 : 45921
Number of testing samples in 2019 : 20667
Number of training samples in 2020 : 45882
Number of testing samples in 2020 : 19894
Features_train shape: (91803, 244, 1)
Labels_train shape: (91803, 976)
Features_test shape: (40561, 244, 1)
Labels_test shape: (40561, 976)
sys.argv n: 8
BATCH_SIZE 1024
Transformer_LSPD_v2_5_new.py
0
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
Train n = 91803


#partical CNN and transformers*************************************************************************




   1: transformer model ********************************************************************************iter
Just position
Is reflectance True
Activation functions is: sigmoid
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 244, 1)]             0         []                            
                                                                                                  
 tf.__operators__.getitem (  (None, 244, 1)               0         ['input_1[0][0]']             
 SlicingOpLambda)                                                                                 
                                                                                                  
 tf.math.not_equal (TFOpLam  (None, 244, 1)               0         ['tf.__operators__.getitem[0][
 bda)                                                               0]']                          
                                                                                                  
 tf.__operators__.getitem_1  (None, 244, 1)               0         ['input_1[0][0]']             
  (SlicingOpLambda)                                                                               
                                                                                                  
 tf.__operators__.getitem_2  (None, 244, 1)               0         ['input_1[0][0]']             
  (SlicingOpLambda)                                                                               
                                                                                                  
 tf.cast (TFOpLambda)        (None, 244, 1)               0         ['tf.math.not_equal[0][0]']   
                                                                                                  
 tf.ones_like (TFOpLambda)   (None, 244, 1)               0         ['tf.__operators__.getitem_1[0
                                                                    ][0]']                        
                                                                                                  
 tf.math.not_equal_1 (TFOpL  (None, 244, 1)               0         ['tf.__operators__.getitem_2[0
 ambda)                                                             ][0]']                        
                                                                                                  
 tf.math.multiply (TFOpLamb  (None, 244, 1)               0         ['tf.__operators__.getitem[0][
 da)                                                                0]',                          
                                                                     'tf.cast[0][0]']             
                                                                                                  
 tf.math.multiply_1 (TFOpLa  (None, 244, 1)               0         ['tf.ones_like[0][0]']        
 mbda)                                                                                            
                                                                                                  
 tf.math.reduce_any (TFOpLa  (None, 244)                  0         ['tf.math.not_equal_1[0][0]'] 
 mbda)                                                                                            
                                                                                                  
 dense (Dense)               (None, 244, 64)              64        ['tf.math.multiply[0][0]']    
                                                                                                  
 dense_1 (Dense)             (None, 244, 64)              64        ['tf.math.multiply_1[0][0]']  
                                                                                                  
 tf.cast_1 (TFOpLambda)      (None, 244)                  0         ['tf.math.reduce_any[0][0]']  
                                                                                                  
 tf.__operators__.add (TFOp  (None, 244, 64)              0         ['dense[0][0]',               
 Lambda)                                                             'dense_1[0][0]']             
                                                                                                  
 tf.__operators__.getitem_3  (None, 1, 1, 244)            0         ['tf.cast_1[0][0]']           
  (SlicingOpLambda)                                                                               
                                                                                                  
 multi_head_attention (Mult  ((None, 244, 64),            16640     ['tf.__operators__.add[0][0]',
 iHeadAttention)              (None, 4, 244, 244))                   'tf.__operators__.getitem_3[0
                                                                    ][0]',                        
                                                                     'tf.__operators__.add[0][0]',
                                                                     'tf.__operators__.add[0][0]']
                                                                                                  
 dropout (Dropout)           (None, 244, 64)              0         ['multi_head_attention[0][0]']
                                                                                                  
 tf.__operators__.add_1 (TF  (None, 244, 64)              0         ['tf.__operators__.add[0][0]',
 OpLambda)                                                           'dropout[0][0]']             
                                                                                                  
 layer_normalization (Layer  (None, 244, 64)              128       ['tf.__operators__.add_1[0][0]
 Normalization)                                                     ']                            
                                                                                                  
 sequential (Sequential)     (None, 244, 64)              33088     ['layer_normalization[0][0]'] 
                                                                                                  
 dropout_1 (Dropout)         (None, 244, 64)              0         ['sequential[0][0]']          
                                                                                                  
 tf.__operators__.add_2 (TF  (None, 244, 64)              0         ['layer_normalization[0][0]', 
 OpLambda)                                                           'dropout_1[0][0]']           
                                                                                                  
 layer_normalization_1 (Lay  (None, 244, 64)              128       ['tf.__operators__.add_2[0][0]
 erNormalization)                                                   ']                            
                                                                                                  
 multi_head_attention_1 (Mu  ((None, 244, 64),            16640     ['layer_normalization_1[0][0]'
 ltiHeadAttention)            (None, 4, 244, 244))                  , 'tf.__operators__.getitem_3[
                                                                    0][0]',                       
                                                                     'layer_normalization_1[0][0]'
                                                                    , 'layer_normalization_1[0][0]
                                                                    ']                            
                                                                                                  
 dropout_2 (Dropout)         (None, 244, 64)              0         ['multi_head_attention_1[0][0]
                                                                    ']                            
                                                                                                  
 tf.__operators__.add_3 (TF  (None, 244, 64)              0         ['layer_normalization_1[0][0]'
 OpLambda)                                                          , 'dropout_2[0][0]']          
                                                                                                  
 layer_normalization_2 (Lay  (None, 244, 64)              128       ['tf.__operators__.add_3[0][0]
 erNormalization)                                                   ']                            
                                                                                                  
 sequential_1 (Sequential)   (None, 244, 64)              33088     ['layer_normalization_2[0][0]'
                                                                    ]                             
                                                                                                  
 dropout_3 (Dropout)         (None, 244, 64)              0         ['sequential_1[0][0]']        
                                                                                                  
 tf.__operators__.add_4 (TF  (None, 244, 64)              0         ['layer_normalization_2[0][0]'
 OpLambda)                                                          , 'dropout_3[0][0]']          
                                                                                                  
 layer_normalization_3 (Lay  (None, 244, 64)              128       ['tf.__operators__.add_4[0][0]
 erNormalization)                                                   ']                            
                                                                                                  
 multi_head_attention_2 (Mu  ((None, 244, 64),            16640     ['layer_normalization_3[0][0]'
 ltiHeadAttention)            (None, 4, 244, 244))                  , 'tf.__operators__.getitem_3[
                                                                    0][0]',                       
                                                                     'layer_normalization_3[0][0]'
                                                                    , 'layer_normalization_3[0][0]
                                                                    ']                            
                                                                                                  
 dropout_4 (Dropout)         (None, 244, 64)              0         ['multi_head_attention_2[0][0]
                                                                    ']                            
                                                                                                  
 tf.__operators__.add_5 (TF  (None, 244, 64)              0         ['layer_normalization_3[0][0]'
 OpLambda)                                                          , 'dropout_4[0][0]']          
                                                                                                  
 layer_normalization_4 (Lay  (None, 244, 64)              128       ['tf.__operators__.add_5[0][0]
 erNormalization)                                                   ']                            
                                                                                                  
 sequential_2 (Sequential)   (None, 244, 64)              33088     ['layer_normalization_4[0][0]'
                                                                    ]                             
                                                                                                  
 dropout_5 (Dropout)         (None, 244, 64)              0         ['sequential_2[0][0]']        
                                                                                                  
 tf.__operators__.add_6 (TF  (None, 244, 64)              0         ['layer_normalization_4[0][0]'
 OpLambda)                                                          , 'dropout_5[0][0]']          
                                                                                                  
 layer_normalization_5 (Lay  (None, 244, 64)              128       ['tf.__operators__.add_6[0][0]
 erNormalization)                                                   ']                            
                                                                                                  
 dense_10 (Dense)            (None, 244, 4)               260       ['layer_normalization_5[0][0]'
                                                                    ]                             
                                                                                                  
==================================================================================================
Total params: 150340 (587.27 KB)
Trainable params: 150340 (587.27 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
Activation functions is: sigmoid
Loss function is MSE
momentum=0.9	learning rate=0.01  decay=0.0001
tfa.optimizers.AdamW 
Epoch 1/70
87/87 - 12s - loss: 0.0180 - root_mean_squared_error: 0.1343 - val_loss: 0.0040 - val_root_mean_squared_error: 0.0632 - lr: 0.0020 - 12s/epoch - 133ms/step
Epoch 2/70
87/87 - 9s - loss: 0.0025 - root_mean_squared_error: 0.0500 - val_loss: 0.0029 - val_root_mean_squared_error: 0.0539 - lr: 0.0039 - 9s/epoch - 105ms/step
Epoch 3/70
87/87 - 9s - loss: 0.0015 - root_mean_squared_error: 0.0387 - val_loss: 0.0025 - val_root_mean_squared_error: 0.0499 - lr: 0.0059 - 9s/epoch - 105ms/step
Epoch 4/70
87/87 - 9s - loss: 0.0013 - root_mean_squared_error: 0.0355 - val_loss: 0.0026 - val_root_mean_squared_error: 0.0508 - lr: 0.0078 - 9s/epoch - 105ms/step
Epoch 5/70
87/87 - 9s - loss: 9.8793e-04 - root_mean_squared_error: 0.0314 - val_loss: 0.0018 - val_root_mean_squared_error: 0.0420 - lr: 0.0098 - 9s/epoch - 105ms/step
Epoch 6/70
87/87 - 9s - loss: 8.6882e-04 - root_mean_squared_error: 0.0295 - val_loss: 0.0014 - val_root_mean_squared_error: 0.0372 - lr: 0.0100 - 9s/epoch - 105ms/step
Epoch 7/70
87/87 - 9s - loss: 7.1750e-04 - root_mean_squared_error: 0.0268 - val_loss: 0.0013 - val_root_mean_squared_error: 0.0363 - lr: 0.0100 - 9s/epoch - 106ms/step
Epoch 8/70
87/87 - 9s - loss: 6.6368e-04 - root_mean_squared_error: 0.0258 - val_loss: 0.0016 - val_root_mean_squared_error: 0.0395 - lr: 0.0100 - 9s/epoch - 106ms/step
Epoch 9/70
87/87 - 9s - loss: 5.8104e-04 - root_mean_squared_error: 0.0241 - val_loss: 0.0021 - val_root_mean_squared_error: 0.0460 - lr: 0.0099 - 9s/epoch - 106ms/step
Epoch 10/70
87/87 - 9s - loss: 5.5450e-04 - root_mean_squared_error: 0.0235 - val_loss: 0.0011 - val_root_mean_squared_error: 0.0325 - lr: 0.0099 - 9s/epoch - 106ms/step
Epoch 11/70
87/87 - 9s - loss: 5.2246e-04 - root_mean_squared_error: 0.0229 - val_loss: 0.0012 - val_root_mean_squared_error: 0.0349 - lr: 0.0099 - 9s/epoch - 106ms/step
Epoch 12/70
87/87 - 9s - loss: 4.7966e-04 - root_mean_squared_error: 0.0219 - val_loss: 0.0012 - val_root_mean_squared_error: 0.0342 - lr: 0.0098 - 9s/epoch - 106ms/step
Epoch 13/70
87/87 - 9s - loss: 4.4310e-04 - root_mean_squared_error: 0.0210 - val_loss: 0.0011 - val_root_mean_squared_error: 0.0335 - lr: 0.0097 - 9s/epoch - 106ms/step
Epoch 14/70
87/87 - 9s - loss: 4.1564e-04 - root_mean_squared_error: 0.0204 - val_loss: 8.0178e-04 - val_root_mean_squared_error: 0.0283 - lr: 0.0097 - 9s/epoch - 106ms/step
Epoch 15/70
87/87 - 9s - loss: 3.9706e-04 - root_mean_squared_error: 0.0199 - val_loss: 8.6555e-04 - val_root_mean_squared_error: 0.0294 - lr: 0.0096 - 9s/epoch - 106ms/step
Epoch 16/70
87/87 - 9s - loss: 3.8455e-04 - root_mean_squared_error: 0.0196 - val_loss: 8.3788e-04 - val_root_mean_squared_error: 0.0289 - lr: 0.0095 - 9s/epoch - 106ms/step
Epoch 17/70
87/87 - 9s - loss: 3.8068e-04 - root_mean_squared_error: 0.0195 - val_loss: 9.4659e-04 - val_root_mean_squared_error: 0.0308 - lr: 0.0094 - 9s/epoch - 106ms/step
Epoch 18/70
87/87 - 9s - loss: 3.6788e-04 - root_mean_squared_error: 0.0192 - val_loss: 9.8243e-04 - val_root_mean_squared_error: 0.0313 - lr: 0.0093 - 9s/epoch - 106ms/step
Epoch 19/70
87/87 - 9s - loss: 3.5818e-04 - root_mean_squared_error: 0.0189 - val_loss: 8.5841e-04 - val_root_mean_squared_error: 0.0293 - lr: 0.0092 - 9s/epoch - 106ms/step
Epoch 20/70
87/87 - 9s - loss: 3.5563e-04 - root_mean_squared_error: 0.0189 - val_loss: 9.1907e-04 - val_root_mean_squared_error: 0.0303 - lr: 0.0091 - 9s/epoch - 106ms/step
Epoch 21/70
87/87 - 9s - loss: 3.6636e-04 - root_mean_squared_error: 0.0191 - val_loss: 9.0424e-04 - val_root_mean_squared_error: 0.0301 - lr: 0.0090 - 9s/epoch - 106ms/step
Epoch 22/70
87/87 - 9s - loss: 3.4370e-04 - root_mean_squared_error: 0.0185 - val_loss: 8.4018e-04 - val_root_mean_squared_error: 0.0290 - lr: 0.0089 - 9s/epoch - 106ms/step
Epoch 23/70
87/87 - 9s - loss: 3.4779e-04 - root_mean_squared_error: 0.0186 - val_loss: 8.5803e-04 - val_root_mean_squared_error: 0.0293 - lr: 0.0087 - 9s/epoch - 106ms/step
Epoch 24/70
87/87 - 9s - loss: 3.3585e-04 - root_mean_squared_error: 0.0183 - val_loss: 7.9940e-04 - val_root_mean_squared_error: 0.0283 - lr: 0.0086 - 9s/epoch - 106ms/step
Epoch 25/70
87/87 - 9s - loss: 3.3462e-04 - root_mean_squared_error: 0.0183 - val_loss: 7.1533e-04 - val_root_mean_squared_error: 0.0267 - lr: 0.0084 - 9s/epoch - 106ms/step
Epoch 26/70
87/87 - 9s - loss: 3.3799e-04 - root_mean_squared_error: 0.0184 - val_loss: 8.2499e-04 - val_root_mean_squared_error: 0.0287 - lr: 0.0083 - 9s/epoch - 106ms/step
Epoch 27/70
87/87 - 9s - loss: 3.2720e-04 - root_mean_squared_error: 0.0181 - val_loss: 7.1650e-04 - val_root_mean_squared_error: 0.0268 - lr: 0.0081 - 9s/epoch - 106ms/step
Epoch 28/70
87/87 - 9s - loss: 3.1611e-04 - root_mean_squared_error: 0.0178 - val_loss: 6.5466e-04 - val_root_mean_squared_error: 0.0256 - lr: 0.0080 - 9s/epoch - 106ms/step
Epoch 29/70
87/87 - 9s - loss: 3.1830e-04 - root_mean_squared_error: 0.0178 - val_loss: 8.1015e-04 - val_root_mean_squared_error: 0.0285 - lr: 0.0078 - 9s/epoch - 106ms/step
Epoch 30/70
87/87 - 9s - loss: 3.2295e-04 - root_mean_squared_error: 0.0180 - val_loss: 6.7470e-04 - val_root_mean_squared_error: 0.0260 - lr: 0.0076 - 9s/epoch - 106ms/step
Epoch 31/70
87/87 - 9s - loss: 3.1433e-04 - root_mean_squared_error: 0.0177 - val_loss: 7.4034e-04 - val_root_mean_squared_error: 0.0272 - lr: 0.0074 - 9s/epoch - 106ms/step
Epoch 32/70
87/87 - 9s - loss: 3.0928e-04 - root_mean_squared_error: 0.0176 - val_loss: 7.3808e-04 - val_root_mean_squared_error: 0.0272 - lr: 0.0073 - 9s/epoch - 106ms/step
Epoch 33/70
87/87 - 9s - loss: 3.0009e-04 - root_mean_squared_error: 0.0173 - val_loss: 7.0182e-04 - val_root_mean_squared_error: 0.0265 - lr: 0.0071 - 9s/epoch - 106ms/step
Epoch 34/70
87/87 - 9s - loss: 3.0217e-04 - root_mean_squared_error: 0.0174 - val_loss: 9.7508e-04 - val_root_mean_squared_error: 0.0312 - lr: 0.0069 - 9s/epoch - 106ms/step
Epoch 35/70
87/87 - 9s - loss: 3.0003e-04 - root_mean_squared_error: 0.0173 - val_loss: 7.9408e-04 - val_root_mean_squared_error: 0.0282 - lr: 0.0067 - 9s/epoch - 106ms/step
Epoch 36/70
87/87 - 9s - loss: 2.9644e-04 - root_mean_squared_error: 0.0172 - val_loss: 6.6011e-04 - val_root_mean_squared_error: 0.0257 - lr: 0.0065 - 9s/epoch - 106ms/step
Epoch 37/70
87/87 - 9s - loss: 2.9413e-04 - root_mean_squared_error: 0.0172 - val_loss: 8.5484e-04 - val_root_mean_squared_error: 0.0292 - lr: 0.0063 - 9s/epoch - 106ms/step
Epoch 38/70
87/87 - 9s - loss: 2.9041e-04 - root_mean_squared_error: 0.0170 - val_loss: 6.5616e-04 - val_root_mean_squared_error: 0.0256 - lr: 0.0061 - 9s/epoch - 106ms/step
Epoch 39/70
87/87 - 9s - loss: 2.8863e-04 - root_mean_squared_error: 0.0170 - val_loss: 7.0190e-04 - val_root_mean_squared_error: 0.0265 - lr: 0.0059 - 9s/epoch - 106ms/step
Epoch 40/70
87/87 - 9s - loss: 2.8992e-04 - root_mean_squared_error: 0.0170 - val_loss: 6.8247e-04 - val_root_mean_squared_error: 0.0261 - lr: 0.0057 - 9s/epoch - 106ms/step
Epoch 41/70
87/87 - 9s - loss: 2.8997e-04 - root_mean_squared_error: 0.0170 - val_loss: 7.2976e-04 - val_root_mean_squared_error: 0.0270 - lr: 0.0055 - 9s/epoch - 106ms/step
Epoch 42/70
87/87 - 9s - loss: 2.8361e-04 - root_mean_squared_error: 0.0168 - val_loss: 7.3553e-04 - val_root_mean_squared_error: 0.0271 - lr: 0.0053 - 9s/epoch - 106ms/step
Epoch 43/70
87/87 - 9s - loss: 2.8412e-04 - root_mean_squared_error: 0.0169 - val_loss: 7.5076e-04 - val_root_mean_squared_error: 0.0274 - lr: 0.0051 - 9s/epoch - 106ms/step
Epoch 44/70
87/87 - 9s - loss: 2.7969e-04 - root_mean_squared_error: 0.0167 - val_loss: 6.2370e-04 - val_root_mean_squared_error: 0.0250 - lr: 0.0049 - 9s/epoch - 106ms/step
Epoch 45/70
87/87 - 9s - loss: 2.7703e-04 - root_mean_squared_error: 0.0166 - val_loss: 7.5708e-04 - val_root_mean_squared_error: 0.0275 - lr: 0.0047 - 9s/epoch - 106ms/step
Epoch 46/70
87/87 - 9s - loss: 2.7398e-04 - root_mean_squared_error: 0.0166 - val_loss: 6.8083e-04 - val_root_mean_squared_error: 0.0261 - lr: 0.0045 - 9s/epoch - 106ms/step
Epoch 47/70
87/87 - 9s - loss: 2.7347e-04 - root_mean_squared_error: 0.0165 - val_loss: 7.7175e-04 - val_root_mean_squared_error: 0.0278 - lr: 0.0043 - 9s/epoch - 106ms/step
Epoch 48/70
87/87 - 9s - loss: 2.7240e-04 - root_mean_squared_error: 0.0165 - val_loss: 7.9973e-04 - val_root_mean_squared_error: 0.0283 - lr: 0.0041 - 9s/epoch - 106ms/step
Epoch 49/70
87/87 - 9s - loss: 2.7359e-04 - root_mean_squared_error: 0.0165 - val_loss: 7.6196e-04 - val_root_mean_squared_error: 0.0276 - lr: 0.0039 - 9s/epoch - 106ms/step
Epoch 50/70
87/87 - 9s - loss: 2.7137e-04 - root_mean_squared_error: 0.0165 - val_loss: 6.2870e-04 - val_root_mean_squared_error: 0.0251 - lr: 0.0037 - 9s/epoch - 106ms/step
Epoch 51/70
87/87 - 9s - loss: 2.6909e-04 - root_mean_squared_error: 0.0164 - val_loss: 7.3833e-04 - val_root_mean_squared_error: 0.0272 - lr: 0.0035 - 9s/epoch - 106ms/step
Epoch 52/70
87/87 - 9s - loss: 2.6838e-04 - root_mean_squared_error: 0.0164 - val_loss: 9.7772e-04 - val_root_mean_squared_error: 0.0313 - lr: 0.0033 - 9s/epoch - 106ms/step
Epoch 53/70
87/87 - 9s - loss: 2.6654e-04 - root_mean_squared_error: 0.0163 - val_loss: 6.7690e-04 - val_root_mean_squared_error: 0.0260 - lr: 0.0031 - 9s/epoch - 106ms/step
Epoch 54/70
87/87 - 9s - loss: 2.6381e-04 - root_mean_squared_error: 0.0162 - val_loss: 7.6040e-04 - val_root_mean_squared_error: 0.0276 - lr: 0.0029 - 9s/epoch - 106ms/step
Epoch 55/70
87/87 - 9s - loss: 2.6688e-04 - root_mean_squared_error: 0.0163 - val_loss: 6.5958e-04 - val_root_mean_squared_error: 0.0257 - lr: 0.0027 - 9s/epoch - 106ms/step
Epoch 56/70
87/87 - 9s - loss: 2.6635e-04 - root_mean_squared_error: 0.0163 - val_loss: 7.7746e-04 - val_root_mean_squared_error: 0.0279 - lr: 0.0025 - 9s/epoch - 106ms/step
Epoch 57/70
87/87 - 9s - loss: 2.6235e-04 - root_mean_squared_error: 0.0162 - val_loss: 8.1097e-04 - val_root_mean_squared_error: 0.0285 - lr: 0.0024 - 9s/epoch - 106ms/step
Epoch 58/70
87/87 - 9s - loss: 2.6036e-04 - root_mean_squared_error: 0.0161 - val_loss: 8.8953e-04 - val_root_mean_squared_error: 0.0298 - lr: 0.0022 - 9s/epoch - 106ms/step
Epoch 59/70
87/87 - 9s - loss: 2.5941e-04 - root_mean_squared_error: 0.0161 - val_loss: 7.1075e-04 - val_root_mean_squared_error: 0.0267 - lr: 0.0020 - 9s/epoch - 106ms/step
Epoch 60/70
87/87 - 9s - loss: 2.5966e-04 - root_mean_squared_error: 0.0161 - val_loss: 9.0653e-04 - val_root_mean_squared_error: 0.0301 - lr: 0.0019 - 9s/epoch - 106ms/step
Epoch 61/70
87/87 - 9s - loss: 2.5935e-04 - root_mean_squared_error: 0.0161 - val_loss: 7.8199e-04 - val_root_mean_squared_error: 0.0280 - lr: 0.0017 - 9s/epoch - 106ms/step
Epoch 62/70
87/87 - 9s - loss: 2.5911e-04 - root_mean_squared_error: 0.0161 - val_loss: 9.2417e-04 - val_root_mean_squared_error: 0.0304 - lr: 0.0016 - 9s/epoch - 106ms/step
Epoch 63/70
87/87 - 9s - loss: 2.5820e-04 - root_mean_squared_error: 0.0161 - val_loss: 7.6949e-04 - val_root_mean_squared_error: 0.0277 - lr: 0.0014 - 9s/epoch - 106ms/step
Epoch 64/70
87/87 - 9s - loss: 2.5734e-04 - root_mean_squared_error: 0.0160 - val_loss: 7.9587e-04 - val_root_mean_squared_error: 0.0282 - lr: 0.0013 - 9s/epoch - 106ms/step
Epoch 65/70
87/87 - 9s - loss: 2.5341e-04 - root_mean_squared_error: 0.0159 - val_loss: 8.4394e-04 - val_root_mean_squared_error: 0.0291 - lr: 0.0011 - 9s/epoch - 106ms/step
Epoch 66/70
87/87 - 9s - loss: 2.5590e-04 - root_mean_squared_error: 0.0160 - val_loss: 8.3532e-04 - val_root_mean_squared_error: 0.0289 - lr: 0.0010 - 9s/epoch - 106ms/step
Epoch 67/70
87/87 - 9s - loss: 2.5298e-04 - root_mean_squared_error: 0.0159 - val_loss: 8.3156e-04 - val_root_mean_squared_error: 0.0288 - lr: 8.9492e-04 - 9s/epoch - 106ms/step
Epoch 68/70
87/87 - 9s - loss: 2.5294e-04 - root_mean_squared_error: 0.0159 - val_loss: 7.8932e-04 - val_root_mean_squared_error: 0.0281 - lr: 7.8152e-04 - 9s/epoch - 106ms/step
Epoch 69/70
87/87 - 9s - loss: 2.5181e-04 - root_mean_squared_error: 0.0159 - val_loss: 8.3426e-04 - val_root_mean_squared_error: 0.0289 - lr: 6.7518e-04 - 9s/epoch - 106ms/step
Epoch 70/70
87/87 - 9s - loss: 2.5134e-04 - root_mean_squared_error: 0.0159 - val_loss: 7.8963e-04 - val_root_mean_squared_error: 0.0281 - lr: 5.7609e-04 - 9s/epoch - 106ms/step
[0.001955056, 0.003910112, 0.0058651688, 0.007820224, 0.009775281, 0.009996717, 0.009985113, 0.009965152, 0.009936867, 0.009900306, 0.009855531, 0.009802615, 0.009741649, 0.009672734, 0.0095959855, 0.009511532, 0.009419516, 0.0093200905, 0.009213423, 0.009099693, 0.00897909, 0.008851817, 0.0087180855, 0.008578122, 0.008432158, 0.008280443, 0.008123227, 0.007960777, 0.0077933627, 0.0076212655, 0.007444774, 0.0072641843, 0.007079799, 0.0068919263, 0.006700883, 0.0065069883, 0.0063105673, 0.0061119483, 0.005911467, 0.0057094563, 0.0055062566, 0.005302208, 0.0050976537, 0.0048929355, 0.0046883957, 0.0044843787, 0.004281226, 0.0040792795, 0.003878875, 0.0036803507, 0.0034840384, 0.0032902681, 0.0030993633, 0.0029116438, 0.0027274268, 0.0025470185, 0.002370723, 0.0021988347, 0.0020316418, 0.0018694263, 0.0017124587, 0.0015610015, 0.0014153096, 0.0012756273, 0.0011421889, 0.0010152167, 0.000894925, 0.0007815152, 0.00067517755, 0.000576089]
   1/1268 [..............................] - ETA: 6:41  16/1268 [..............................] - ETA: 4s    31/1268 [..............................] - ETA: 4s  46/1268 [>.............................] - ETA: 4s  61/1268 [>.............................] - ETA: 4s  76/1268 [>.............................] - ETA: 4s  91/1268 [=>............................] - ETA: 4s 106/1268 [=>............................] - ETA: 3s 121/1268 [=>............................] - ETA: 3s 136/1268 [==>...........................] - ETA: 3s 151/1268 [==>...........................] - ETA: 3s 166/1268 [==>...........................] - ETA: 3s 181/1268 [===>..........................] - ETA: 3s 196/1268 [===>..........................] - ETA: 3s 211/1268 [===>..........................] - ETA: 3s 226/1268 [====>.........................] - ETA: 3s 241/1268 [====>.........................] - ETA: 3s 256/1268 [=====>........................] - ETA: 3s 271/1268 [=====>........................] - ETA: 3s 286/1268 [=====>........................] - ETA: 3s 301/1268 [======>.......................] - ETA: 3s 316/1268 [======>.......................] - ETA: 3s 331/1268 [======>.......................] - ETA: 3s 346/1268 [=======>......................] - ETA: 3s 361/1268 [=======>......................] - ETA: 3s 376/1268 [=======>......................] - ETA: 3s 391/1268 [========>.....................] - ETA: 3s 406/1268 [========>.....................] - ETA: 2s 421/1268 [========>.....................] - ETA: 2s 436/1268 [=========>....................] - ETA: 2s 451/1268 [=========>....................] - ETA: 2s 466/1268 [==========>...................] - ETA: 2s 481/1268 [==========>...................] - ETA: 2s 496/1268 [==========>...................] - ETA: 2s 511/1268 [===========>..................] - ETA: 2s 526/1268 [===========>..................] - ETA: 2s 541/1268 [===========>..................] - ETA: 2s 556/1268 [============>.................] - ETA: 2s 571/1268 [============>.................] - ETA: 2s 586/1268 [============>.................] - ETA: 2s 601/1268 [=============>................] - ETA: 2s 616/1268 [=============>................] - ETA: 2s 631/1268 [=============>................] - ETA: 2s 646/1268 [==============>...............] - ETA: 2s 661/1268 [==============>...............] - ETA: 2s 676/1268 [==============>...............] - ETA: 2s 691/1268 [===============>..............] - ETA: 1s 706/1268 [===============>..............] - ETA: 1s 721/1268 [================>.............] - ETA: 1s 736/1268 [================>.............] - ETA: 1s 751/1268 [================>.............] - ETA: 1s 766/1268 [=================>............] - ETA: 1s 781/1268 [=================>............] - ETA: 1s 796/1268 [=================>............] - ETA: 1s 811/1268 [==================>...........] - ETA: 1s 826/1268 [==================>...........] - ETA: 1s 841/1268 [==================>...........] - ETA: 1s 856/1268 [===================>..........] - ETA: 1s 871/1268 [===================>..........] - ETA: 1s 886/1268 [===================>..........] - ETA: 1s 901/1268 [====================>.........] - ETA: 1s 916/1268 [====================>.........] - ETA: 1s 931/1268 [=====================>........] - ETA: 1s 946/1268 [=====================>........] - ETA: 1s 961/1268 [=====================>........] - ETA: 1s 976/1268 [======================>.......] - ETA: 0s 991/1268 [======================>.......] - ETA: 0s1006/1268 [======================>.......] - ETA: 0s1021/1268 [=======================>......] - ETA: 0s1036/1268 [=======================>......] - ETA: 0s1051/1268 [=======================>......] - ETA: 0s1066/1268 [========================>.....] - ETA: 0s1081/1268 [========================>.....] - ETA: 0s1096/1268 [========================>.....] - ETA: 0s1111/1268 [=========================>....] - ETA: 0s1126/1268 [=========================>....] - ETA: 0s1141/1268 [=========================>....] - ETA: 0s1156/1268 [==========================>...] - ETA: 0s1171/1268 [==========================>...] - ETA: 0s1186/1268 [===========================>..] - ETA: 0s1201/1268 [===========================>..] - ETA: 0s1216/1268 [===========================>..] - ETA: 0s1231/1268 [============================>.] - ETA: 0s1246/1268 [============================>.] - ETA: 0s1261/1268 [============================>.] - ETA: 0s1268/1268 [==============================] - 5s 3ms/step
>>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics Pearson_R: [0.84 0.88 0.73 0.79]
>>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics p_value:   [0. 0. 0. 0.]
>>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics MAD:       [12.5  9.4 14.7 14.7]
>>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics RMSE:      [18.4 13.7 21.  21. ]
>>>>>>>>>>>>>>>tranfatt: accuracy for four key phenometrics MSB:       [-3.1  0.1  0.5 -5.3]