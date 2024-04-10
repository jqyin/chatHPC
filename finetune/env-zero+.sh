module load rocm/5.7.0
BUILD=/lustre/orion/scratch/junqi/stf218/zero++/miniconda
export PATH=${BUILD}/bin:$PATH
source ${BUILD}/etc/profile.d/conda.sh


conda activate /lustre/orion/scratch/junqi/stf218/zero++/miniconda/envs/zero+-env
export LD_LIBRARY_PATH=/lustre/orion/scratch/junqi/stf218/pyt-nightly-rocm5.7/rccl-plugin-rocm570/lib:$LD_LIBRARY_PATH
