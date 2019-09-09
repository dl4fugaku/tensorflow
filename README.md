```
export TFBASE=/tmp
cd ${TFBASE}

git clone https://github.com/spack/spack.git
. ${TFBASE}/spack/share/spack/setup-env.sh
spack compiler find
spack install gcc@9.1.0
spack load gcc@9.1.0
spack compiler find
spack install git%gcc@9.1.0
spack load git
spack install hdf5@1.10.5%gcc@9.1.0
spack load hdf5
spack install openblas@0.3.6%gcc@9.1.0 threads=openmp
spack load openblas
spack install python@3.7.3%gcc@9.1.0 ^ncurses@6.1 ^pkgconf@1.6.1 ^xz@5.2.4 ^readline@7.0
spack load python@3.7.3
spack install py-setuptools ^python@3.7.3
spack load py-setuptools
spack install py-numpy@1.16.4 ^openblas@0.3.6 /gyvjlof ^py-setuptools@41.0.1 ^python@3.7.3
spack load py-numpy
spack install py-scipy@1.2.1 ^py-numpy@1.16.4 ^openblas@0.3.6 /gyvjlof ^py-setuptools@41.0.1 ^python@3.7.3
spack load py-scipy
spack install py-pip ^python@3.7.3
spack install py-wheel ^python@3.7.3
spack install py-pyyaml ^python@3.7.3
spack install py-six ^python@3.7.3
spack install py-pbr ^python@3.7.3
spack install py-future ^python@3.7.3
spack install py-mock ^python@3.7.3
spack install py-theano ^py-numpy@1.16.4 ^openblas@0.3.6 /gyvjlof ^python@3.7.3
spack install py-keras-preprocessing ^python@3.7.3
spack install py-keras-applications ^python@3.7.3
spack install py-keras ^openblas@0.3.6 /gyvjlof ^python@3.7.3
spack load py-pip; spack load py-wheel; spack load py-pyyaml; spack load py-six; spack load py-pbr; spack load py-future; spack load py-mock;
spack load py-theano; spack load py-keras-preprocessing; spack load py-keras-applications; spack load py-keras

PYPATH=`which python`
PY2PATH=`dirname ${PYPATH}`
PY2PATH=`dirname ${PY2PATH}`
cd ${PY2PATH}
for x in setuptools pip scipy mock wheel future theano keras numpy pbr; do ln -s ../../../../py-${x}-*/lib/python3.7/site-packages/${x} .; done
ln -s ../../../../py-pyyaml-*/lib/python3.7/site-packages/yaml .
ln -s ../../../../py-keras-applications-*/lib/python3.7/site-packages/keras_applications .
ln -s ../../../../py-keras-preprocessing-*/lib/python3.7/site-packages/keras_preprocessing .

cd ${TFBASE}
wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-dist.zip
mkdir bazel-0.25.2; cd bazel-0.25.2
unzip ../bazel-0.25.2-dist.zip
PROTOC=`pwd`/bin/protoc EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" GRPC_JAVA_PLUGIN=`pwd`/grpc-java bash ./compile.sh
export PATH=$PATH:`pwd`/../bazel-0.25.2/output

cd ${TFBASE}
git clone https://github.com/dl4fugaku/tensorflow.git
cd ./tensorflow
git checkout fugaku-v2rc0-blas
BLAS_INSTALL_PATH=`echo ${TFBASE}/spack/opt/spack/linux-*/gcc-9.1.0/openblas-*`
BLAS_LIB_A=libopenblas.a
echo "${PYPATH}\n${PY2PATH}/lib/python3.7/site-packages\nn\nn\nn\nn\nn\nn\n\nn" | TEST_TMPDIR=../.cache/bazel ./configure
sed -i -e "s#/scr0/jens#${TFBASE}#" ./WORKSPACE
sed -i -e "s#/scr0/jens#${TFBASE}#" ./tensorflow/core/kernels/gemm_functors.h
TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --output_base=${TFBASE}/log build \
	-s --define=tensorflow_mkldnn_contraction_kernel=0 --config=numa --config=v2 \
	--config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl \
	--copt=-march=native --copt=-Ofast --copt=-finline-functions --copt=-findirect-inlining \
	--cxxopt=-march=native --cxxopt=-Ofast --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
	--copt=-DUSE_CBLAS_GEMM --cxxopt=-DUSE_CBLAS_GEMM \
	--copt=-I${BLAS_INSTALL_PATH}/include --linkopt=-L${BLAS_INSTALL_PATH}/lib --linkopt=-l:${BLAS_LIB_A} \
	//tensorflow/tools/pip_package:build_pip_package \
	--use_action_cache --verbose_failures --repository_cache=${TFBASE}/.cache/bazel --disk_cache=${TFBASE}/.cache/bazel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg
python3.7 -m pip install --upgrade ./tmp/tensorflow_pkg/tensorflow-2.0.0rc0-cp37-cp37m-linux_x86_64.whl
```


⇒  OMP_NUM_THREADS=18 GOMP_SPINCOUNT=0 OMP_SCHEDULE=static OMP_DISPLAY_ENV=TRUE TF_NUM_INTEROP_THREADS=18 TF_NUM_INTRAOP_THREADS=2 python3.7 -m benchmarker  --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4                                                                                                                                      
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
WARNING: Logging before flag parsing goes to stderr.
W0909 18:33:11.245654 139963830617920 deprecation.py:506] From /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
preheat
W0909 18:33:17.685415 139963830617920 deprecation.py:323] From /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Epoch 1/1
32/32 [==============================] - 13s 398ms/step - loss: 1.0923 - acc: 0.8750
train
Epoch 1/3
32/32 [==============================] - 4s 113ms/step - loss: 1.0996 - acc: 0.8750
Epoch 2/3
32/32 [==============================] - 3s 108ms/step - loss: 0.1507 - acc: 0.9062
Epoch 3/3
32/32 [==============================] - 3s 108ms/step - loss: 0.7328 - acc: 0.7500
{
    "batch_size": 4,
    "batch_size_per_device": 4,
    "channels_first": false,
    "cnt_classes": 1000,
    "device": "Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz",
    "framework": "tensorflow",
    "framework_full": "Keras-2.2.4/tensorflow_2.0.0-rc0",
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
            "clock": 2786.048777777776,
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
                "memory_clock": 3004000,
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
    "samples_per_second": 9.114960999922497,
    "time": 3.510711675044149,
    "time_epoch": 3.510711675044149
}

