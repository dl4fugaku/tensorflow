dont use --config=ngraph, its deprecated https://github.com/tensorflow/tensorflow/issues/28470


step 1: install our TF plain version !!!! NEEDS XLA
step 1.1:  need libtensorflow_cc.so
 TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --batch build --config=numa --config=v2 --config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl --copt=-march=native --copt=-Ofast --copt=-finline-functions --copt=-findirect-inlining --cxxopt=-march=native --cxxopt=-Ofast --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow_cc.so --local_cpu_resources=36 --local_ram_resources=$((100*1024)) --use_action_cache --verbose_failures --repository_cache=/scr0/jens/.cache/bazel --disk_cache=/scr0/jens/.cache/bazel

or

TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --batch build --config=numa --config=v2 --config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl  --copt=-march=native --copt=-O3 --copt=-finline-functions --copt=-findirect-inlining --copt=-Wl,--as-needed --cxxopt=-march=native --cxxopt=-O3 --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-Wl,--as-needed --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow_cc.so --local_cpu_resources=$((2*36)) --local_ram_resources=$((100*1024)) --use_action_cache --verbose_failures --repository_cache=/scr0/jens/.cache/bazel --disk_cache=/scr0/jens/.cache/bazel

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




⇒  OMP_NUM_THREADS=18 GOMP_SPINCOUNT=0 OMP_SCHEDULE=static OMP_DISPLAY_ENV=TRUE TF_NUM_INTEROP_THREADS=18 TF_NUM_INTRAOP_THREADS=2 KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,compact,1,0 NGRAPH_INTER_OP_PARALLELISM=18 NGRAPH_INTRA_OP_PARALLELISM=2 NGRAPH_CPU_EIGEN_THREAD_COUNT=18 TF_XLA_FLAGS="--tf_xla_auto_jit=1 --tf_xla_cpu_global_jit" python3.7 -m benchmarker  --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4

OPENMP DISPLAY ENVIRONMENT BEGIN
  _OPENMP = '201511'
  OMP_DYNAMIC = 'FALSE'
  OMP_NESTED = 'FALSE'
  OMP_NUM_THREADS = '18'
  OMP_SCHEDULE = 'STATIC'
  OMP_PROC_BIND = 'FALSE'
  OMP_PLACES = ''
  OMP_STACKSIZE = '0'
  OMP_WAIT_POLICY = 'PASSIVE'
  OMP_THREAD_LIMIT = '4294967295'
  OMP_MAX_ACTIVE_LEVELS = '2147483647'
  OMP_CANCELLATION = 'FALSE'
  OMP_DEFAULT_DEVICE = '0'
  OMP_MAX_TASK_PRIORITY = '0'
  OMP_DISPLAY_AFFINITY = 'FALSE'
  OMP_AFFINITY_FORMAT = 'level %L thread %i affinity %A'
OPENMP DISPLAY ENVIRONMENT END
Using TensorFlow backend.
2019-09-09 17:21:39.662901: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294435000 Hz
2019-09-09 17:21:39.665181: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x48fdf20 executing computations on platform Host. Devices:
2019-09-09 17:21:39.665222: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-09-09 17:21:39.674544: I /scr0/jens/ngraph-bridge/ngraph_bridge/enable_variable_ops/ngraph_rewrite_pass.cc:276] NGraph using backend: CPU
2019-09-09 17:21:39.684450: I /scr0/jens/ngraph-bridge/ngraph_bridge/enable_variable_ops/ngraph_rewrite_pass.cc:276] NGraph using backend: CPU
preheat
WARNING: Logging before flag parsing goes to stderr.
W0909 17:21:46.217470 140554317027136 deprecation.py:323] From /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/lib/python3.7/site-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Epoch 1/1
2019-09-09 17:21:58.347528: I /scr0/jens/ngraph-bridge/ngraph_bridge/enable_variable_ops/ngraph_rewrite_pass.cc:276] NGraph using backend: CPU
2019-09-09 17:22:00.176461: I /scr0/jens/ngraph-bridge/ngraph_bridge/enable_variable_ops/ngraph_rewrite_pass.cc:276] NGraph using backend: CPU
2019-09-09 17:22:04.028774: I /scr0/jens/ngraph-bridge/ngraph_bridge/enable_variable_ops/ngraph_rewrite_pass.cc:276] NGraph using backend: CPU
32/32 [==============================] - 13s 392ms/step - loss: 1.0307 - acc: 0.7500
train
Epoch 1/3
32/32 [==============================] - 4s 117ms/step - loss: 2.1178e-04 - acc: 1.0000
Epoch 2/3
32/32 [==============================] - 4s 117ms/step - loss: 2.0152 - acc: 0.8750
Epoch 3/3
32/32 [==============================] - 4s 115ms/step - loss: 2.0152 - acc: 0.8750
{  
    "batch_size": 4,
    "batch_size_per_device": 4,
    "channels_first": false,
    "cnt_classes": 1000,
    "device": "Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz",
    "framework": "tensorflow",
    "framework_full": "Keras-2.2.4/tensorflow_2.0.0-beta1",
    "gpus": [],
    "misc": null,
    "mode": "training",
    "nb_gpus": 0,
    "path_out": "./logs/training",
    "platform": {
        "cpu": {
            "brand": "Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz",
            "cache": {
                "1": 32768,
                "2": 262144,
                "3": 47185920
            },
            "clock": 2800.3481111111096,
            "clock_max": 3600.0,
            "clock_min": 1200.0,
            "logical_cores": 36,
            "physical_cores": 18
        },
        "gpus": [
            {
                "brand": "Tesla K40c",
                "clock": 745000,
                "compute_capability": 3.5,
                "cores": 2880,
                "memory": 11996954624,
                "multiprocessors": 15,
                "warp_size": 32
            },
            {
                "brand": "Tesla K40c",
                "clock": 745000,
                "compute_capability": 3.5,
                "cores": 2880,
                "memory": 11996954624,
                "memory_clock": 3004000,
                "multiprocessors": 15,
                "warp_size": 32
            }
        ],
        "hdds": {
            "/dev/sda": {
                "model": "Crucial_CT512MX1",
                "size": 1000215216
            },
            "/dev/sdb": {
                "model": "SAMSUNG MZHPU512",
                "size": 1000215216
            }
        },
        "host": "paris0.m.gsic.titech.ac.jp",
        "os": "Linux-3.10.0-957.12.1.el7.x86_64-x86_64-with-centos-7.6.1810-Core",
        "ram": {
            "total": 134941593600
        },
        "swap": 0
    },
    "problem": {
        "bytes_x_train": 19267584,
        "cnt_batches_per_epoch": 8.0,
        "name": "resnet50",
        "shape_x_train": [
            32,
            224,
            224,
            3
        ],
        "size": 32
    },
    "samples_per_second": 8.599896703382974,
    "time": 3.7209749260606864,
    "time_epoch": 3.7209749260606864
}  

