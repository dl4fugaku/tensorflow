step 1: install our TF plain version !!!! NEEDS XLA
step 1.1:  need libtensorflow_cc.so
 TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --batch build --config=numa --config=v2 --config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl --copt=-march=native --copt=-Ofast --copt=-finline-functions --copt=-findirect-inlining --cxxopt=-march=native --cxxopt=-Ofast --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow_cc.so --local_cpu_resources=36 --local_ram_resources=$((100*1024)) --use_action_cache --verbose_failures --repository_cache=/scr0/jens/.cache/bazel --disk_cache=/scr0/jens/.cache/bazel

step 2:
cd /scr0/jens/
git clone https://github.com/tensorflow/ngraph-bridge.git
cd ngraph-bridge/
git checkout -b fugaku-v2b1-ngraph v0.18.0-rc0

for x in `grep -r libtensorflow_framework.so.1 . | cut -d ':' -f1`; do sed -i -e 's/libtensorflow_framework.so.1/libtensorflow_framework.so.2/' $x ; done
for x in `grep -r libtensorflow_cc.so.1 . | cut -d ':' -f1`; do sed -i -e 's/libtensorflow_cc.so.1/libtensorflow_cc.so.2/' $x ; done

mkdir build
cd build

cmake -DNGRAPH_TARGET_ARCH=native -DNGRAPH_TUNE_ARCH=native -DNGRAPH_USE_CXX_ABI=0 -DNGRAPH_DEBUG_ENABLE=NO -DNGRAPH_DISTRIBUTED_ENABLE=OFF -DTF_SRC_DIR=/scr0/jens/tensorflow -DNGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS=TRUE -DNGRAPH_TF_USE_GRAPPLER_OPTIMIZER=TRUE ..
make VERBOSE=1 -j
make install
python3.7 -m pip install --upgrade ./python/dist/ngraph_tensorflow_bridge-0.18.0rc0-py2.py3-none-manylinux1_x86_64.whl
python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__); import ngraph_bridge; print(ngraph_bridge.__version__)"

step3: add ngraph to benchmarker
sed -i '/import keras.models/a import ngraph_bridge' ./benchmarker/modules/_keras.py




// notes
some env stuff for debugging
NGRAPH_TF_VLOG_LEVEL=5
TF_XLA_FLAGS="--tf_xla_auto_jit=1 --tf_xla_cpu_global_jit"
KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,compact,1,0
OMP_NUM_THREADS=18
TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=18
#fucking mess... NGRAPH_INTER_OP_PARALLELISM=2 NGRAPH_INTRA_OP_PARALLELISM=18 NGRAPH_CPU_EIGEN_THREAD_COUNT=18

!!!!! nothing gets faster!! 

alternative test cases:
https://github.com/dragen1860/TensorFlow-2.x-Tutorials/blob/master/08-ResNet/main.py
