#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=60GB
#SBATCH --job-name=label_generation
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mail-user=alnah005@umn.edu
#SBATCH -p amdsmall                                            

cd $SLURM_SUBMIT_DIR
module load singularity
pwd
# singularity exec --nv -i ~/aggregation_for_caesar/aggregation_for_caesar.sif ~/aggregation_for_caesar/commands_solarjets.sh
# singularity exec --nv -i ~/aggregation_for_caesar/aggregation_for_caesar.sif ~/aggregation_for_caesar/commands_preprocessing_gold_standard.sh
singularity exec --nv -i ~/aggregation_for_caesar/aggregation_for_caesar.sif ~/aggregation_for_caesar/commands_preprocessing.sh