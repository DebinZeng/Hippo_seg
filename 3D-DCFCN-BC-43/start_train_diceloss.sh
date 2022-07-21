#export LD_LIBRARY_PATH=/usr/local/cuda/lib64

CAFFE=/home/omnisky/software/3D-Caffe_crfasrnn/build/tools/caffe

for i in 1 2 3 4 5; do
    SNAPSHOT=snapshot_$i/  
    LOGDIR=./training_log_$i/
    mkdir $SNAPSHOT
    mkdir $LOGDIR

    SOLVER=./solver_densenet_$i.prototxt
    
    GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -gpu 1
done


