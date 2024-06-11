#!/bin/bash
#SBATCH --job-name=lsp_transf # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem 507904M # in unit MB 786432M=768G; 785408M=767G (no); 749568M=732G; 131072=128G; 507904M max of innovator
#SBATCH --gres=gpu:2   # Number of GPUs per node (MAX is 2 for SDSU tested on Hank on Apr 19 2021)
#SBATCH --ntasks-per-node=48 # CPUs per node (MAX=48)
#SBATCH --output=log/out-%j-%N.log # Standard output (log file)
#SBATCH --partition=gpu # Partition/Queue
#SBATCH --time=5-00:00:00 # Maximum walltime 192=8-days
##SBATCH --mail-type=ALL #see https://slurm.schedmd.com/sbatch.html for all options. 
##SBATCH --mail-user=khuong.tran@jacks.sdstate.edu

# cd /gpfs/home/hankui.zhang/mycode.research/LCMAP/LCMAP_run
# sbatch my.transformer.gpu.slurm
# squeue -l | grep hankui

# sleep 600
# bash my.transformer.gpu.slurm 11_90
# bash my.transformer.gpu.slurm 11_91
# bash my.transformer.gpu.slurm 11_92
# bash my.transformer.gpu.slurm 11_93
# bash my.transformer.gpu.slurm 11_94

# bash my.transformer.gpu.slurm 11_95
# bash my.transformer.gpu.slurm 11_96

# bash my.transformer.gpu.slurm 11_97


# nvidia-smi


# module load cuda
# # module load cudnn
# module load cudnn/8.2.2.26
# module load python/3.7
# module load rasterionew
# module load libtiff 
# module load libgeotiff

module load rasterio/tensorflow

# module load rasterio/1.3.9

# conda init bash 
date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
which python

rm -rf ./tmp/tb_log/*


SLEEP=60

# version=1_0 #EVI2 features and one-cycle labels + try with random split and spatial split
# version=2_0 #EVI2 features and + try with random split and spatial split + labels distances to onsets

MAXGPU=2; version=2_1; epochs=80 # pre-training 

[ -n "$1" ] && version=${1}

date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;

gpui=0
layer=4
year=1985
year=" "
# for learning_rate in 0.1 0.01 0.001 0.0001;
# for learning_rate in 0.01; ## cnn models 
for learning_rate in 0.01;  ## transformer models 
do
# for L2 in 1e-4 1e-3;
# for L2 in 1e-3 1e-4 1e-5;
for L2 in 1e-4;
do
# for epoch in 100;
for epoch in 80;
# for epoch in ${epochs};
do
for year in " ";
# for year in 2018 2006;
# for year in " " 1985 2006 2018;
do	
#for BATCH in 8 16 32 64 128 512 2048 4096;
for BATCH in 1024;
do
# for DROP in 0 0.1 0.2 0.3 0.4 0.5;
for DROP in 0.1;
# for DROP in 0.25 0.5;
# for DROP in 0.05 0.1;
do
# for version in 2_6;
# for version in prediction;
for version in 2_5_new;
# for version in 12_46;
do
	# learning_rate=0.0001; method=1; epoch=70; L2=0; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	# learning_rate=0.0001; method=2; L2=1e-5; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	method=2; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	echo "python Transformer_LSPD_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${BATCH} 0 ${year} "
	# python  Pro_lcmap_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${BATCH} 0 ${year} > B${BATCH}.rate${learning_rate}.e${epoch}.L${L2}.v${version}.y${year} & 
	python  Transformer_LSPD_v${version}.py ${DROP} ${epoch} ${method} ${learning_rate} ${L2} ${BATCH} ${gpui} ${year} > DROP${DROP}.rate${learning_rate}.b${BATCH}.e${epoch}.L${L2}.v${version}.y${year} & 
	gpui=$((${gpui}+1))
	if [ ${gpui} -ge ${MAXGPU} ]; then 
		wait 
		gpui=0
		echo "gpui=0"
	fi
done 
done 
done
done 
done 
done 
done

wait 
date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
date
echo "$time_diff hours used";



