#!/bin/bash

#SBATCH --time=0-6:00
#SBATCH --account=def-alister
#SBATCH --mem-per-cpu=100GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=run_pnet
#SBATCH --output="run_pnet_jobID-%j.out"

echo 'SETTING UP POINT NET TRAINING ENVIRONMENT'
echo''
echo 'SLURM_TMPDIR'
echo $SLURM_TMPDIR
echo 'slurm job number'
echo $SLURM_JOB_ID
echo ''

cd $SLURM_TMPDIR
echo "current directory: "
pwd
echo ''

mkdir train_data
mkdir scripts
mkdir container
mkdir results

cp /home/russbate/projects/def-alister/russbate/segmentation/containers/baseml_tf_v0.1.37.sif container

cp /home/russbate/projects/def-alister/russbate/segmentation/pnet_data/*.npy train_data

cp /home/russbate/projects/def-alister/russbate/segmentation/python_scripts/* scripts

cd train_data
echo 'training data directory'
ls -lah
echo ''

cd ../scripts
echo 'executables and python scripts'
ls -lah
echo ''

cd ../container
echo 'container'
ls -lh
echo ''

cd ..

echo 'NVIDIA'
nvidia-smi
echo ''

module load singularity/3.8

singularity exec --nv -B /home/russbate/projects/def-alister/russbate -B ${SLURM_TMPDIR} ${SLURM_TMPDIR}/container/baseml_tf_v0.1.37.sif /bin/bash -c "cd ${SLURM_TMPDIR} && python scripts/pnet_train_loop.py >& scripts/pnet_training_job_${SLURM_JOB_ID}.txt" 

echo 'finished singularity exec'
echo ''
ls -lah

cp scripts/pnet_training_job_${SLURM_JOB_ID}.txt /home/russbate/projects/def-alister/russbate/segmentation/training_output/

cp results/* /home/russbate/projects/def-alister/russbate/segmentation/training_output

echo ''
echo 'Finished Job!'
echo ''
