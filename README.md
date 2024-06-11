Code for the publication "A transformer-based model for detecting land surface phenology from the Harmonized Landsat and Sentinel-2 time series across the United States" Remote Sensing of Environment, June 2024, in review.

This code is Copyright Mr. Khuong Tran, South Dakota State University. Use of this code for commercial purposes is not permitted. Contact khuong.tran@jacks.sdstate.edu for more information and updates of this code

Version 1.0 1 Mar 2020
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The code present in the directory consists of

1) Main Python scripts:
- Transformer_LSPD_v2_5_new.py: data preprocessing and normalization; Call Transformer-based model
- transformer_encoder44_v2.py: Transformer-based model
- customized_train_v2.py
- train_test_v2.py
  
2) my.transformer.gpu.slurm:
  This is slurm file to submit job to the HPC (Linux system)

3) v2_5_new.py.yearall.layer3.dim80.METHOD2.DROP0.1.B1024.LR0.01.L20.0001.model.ITER3.h5
  This is h5 of our Transformer-based model

4) DROP0.1.rate0.01.b1024.e80.L1e-4.v2_5_new.y
   This is log file of our model and accuracy printing
