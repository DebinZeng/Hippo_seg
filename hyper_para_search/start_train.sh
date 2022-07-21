#export LD_LIBRARY_PATH=/usr/local/cuda/lib64
LOGDIR=./training_log_Dense
CAFFE=/home/omnisky/software/3D-Caffe/build/tools/caffe
SOLVER=./solver.prototxt
mkdir snapshot_Dense
mkdir $LOGDIR

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -gpu 1


