cd /scr0/jens
git clone https://github.com/spack/spack.git
. /scr0/jens/spack/share/spack/setup-env.sh
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
spack load py-pip; spack load py-wheel; spack load py-pyyaml; spack load py-six; spack load py-pbr; spack load py-future; spack load py-mock
spack load py-theano; spack load py-keras-preprocessing; spack load py-keras-applications; spack load py-keras

cd /scr0/jens
wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-dist.zip
mkdir bazel-0.25.2; cd bazel-0.25.2
unzip ../bazel-0.25.2-dist.zip
PROTOC=`pwd`/bin/protoc EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" GRPC_JAVA_PLUGIN=`pwd`/grpc-java bash ./compile.sh

git clone https://github.com/dl4fugaku/tensorflow.git
cd /scr0/jens/tensorflow/
git checkout -b fugaku-v2b1-plain v2.0.0-beta1
export PATH=$PATH:`pwd`/../bazel-0.25.2/output

TEST_TMPDIR=../.cache/bazel ./configure
$TEST_TMPDIR defined: output root default is '/scr0/jens/tensorflow/../.cache/bazel' and max_idle_secs default is '15'.
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.25.2- (@non-git) installed.
Please specify the location of python. [Default is /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/bin/python]:


Found possible Python library paths:
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-wheel-0.33.1-7jdnl54modjcglz5vx2emxan64a5okmi/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-pyyaml-3.13-dznqm5hsdbz5byq4pk5w3fnhxaiycuiy/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-future-0.16.0-kwly7ba4cr2ufwjlp6wr3msf62ifiv22/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-theano-1.0.4-rgt4yrggrfr7fem5gtzlemm4v3jevy7e/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-mock-2.0.0-utsdmpayrl3pbel2cbxpludjecvhgdsm/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-keras-2.2.4-5btjtcvg66icnznhrt7mnzcrswoenmc3/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-keras-preprocessing-1.0.9-6ppoe4dewroqob74zjazwiblnnwyduaz/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-numpy-1.16.4-qktlhbnwbc22gc6gxzqhcjen5atdyqur/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-scipy-1.2.1-q2zbo67cdnh6zc2ccjtwpp2ms4o5tete/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-setuptools-41.0.1-4f5j3zzyw74pzvxqaojzprooofmqriea/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-keras-applications-1.0.7-vkzl2emt7q4qhljyflx44fiyej3stvfd/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-pip-19.0.3-epdedu4mxtnekbkmlf3vk36n5txqh3lm/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-six-1.12.0-tksvorruxgnu3wt47kyk5l6vwsxxf77n/lib/python3.7/site-packages
  /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-pbr-3.1.1-neuislaevxrfvekpsjsq4ajce3mmt2jf/lib/python3.7/site-packages
Please input the desired Python library path to use.  Default is [/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-wheel-0.33.1-7jdnl54modjcglz5vx2emxan64a5okmi/lib/python3.7/site-packages]
/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/lib/python3.7/site-packages
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
        --config=gdr            # Build with GDR support.
        --config=verbs          # Build with libverbs support.
        --config=ngraph         # Build with Intel nGraph support.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=noaws          # Disable AWS S3 filesystem support.
        --config=nogcp          # Disable GCP support.
        --config=nohdfs         # Disable HDFS support.
        --config=noignite       # Disable Apache Ignite support.
        --config=nokafka        # Disable Apache Kafka support.
        --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished

cd /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-*/lib/python3.7/site-packages
for x in setuptools pip scipy mock wheel future theano keras numpy pbr; do ln -s ../../../../py-${x}-*/lib/python3.7/site-packages/${x} .; done
ln -s ../../../../py-pyyaml-*/lib/python3.7/site-packages/yaml .
ln -s ../../../../py-keras-applications-*/lib/python3.7/site-packages/keras_applications .
ln -s ../../../../py-keras-preprocessing-*/lib/python3.7/site-packages/keras_preprocessing .
TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --batch build --config=mkl --config=numa --config=v2 --config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl --copt=-march=native --copt=-O3 --copt=-finline-functions --copt=-findirect-inlining --copt=-DEIGEN_USE_VML --cxxopt=-march=native --cxxopt=-O3 --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-DEIGEN_USE_VML --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package --local_cpu_resources=36 --local_ram_resources=$((100*1024)) --use_action_cache --verbose_failures --repository_cache=/scr0/jens/.cache/bazel --disk_cache=/scr0/jens/.cache/bazel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg
python3.7 -m pip install ./tmp/tensorflow_pkg/tensorflow-2.0.0b1-cp37-cp37m-linux_x86_64.whl

cd /scr0/jens
git clone --recursive https://github.com/undertherain/benchmarker.git
cd benchmarker/
pip3.7 install system-query
#mod to use tensorflow.compat.v1
#mod /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-keras-2.2.4-5btjtcvg66icnznhrt7mnzcrswoenmc3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py
#mod /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/py-keras-2.2.4-5btjtcvg66icnznhrt7mnzcrswoenmc3/lib/python3.7/site-packages/keras/optimizers.py

KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=18 numactl --physcpubind=0-35 python3.7 -m benchmarker  --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4
Using TensorFlow backend.
2019-07-30 12:32:42.585174: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 2 1 18. Tune using inter_op_parallelism_threads for
 best performance.
preheat
WARNING: Logging before flag parsing goes to stderr.
W0730 12:32:49.256102 140694764250944 deprecation.py:323] From /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.3-civ2iiqopzfrzuhnit4hpexjlm4uwp5b/lib/python3.7/site-pac
kages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Epoch 1/1
32/32 [==============================] - 21s 670ms/step - loss: 1.1326 - acc: 0.8750
train
Epoch 1/3
32/32 [==============================] - 2s 74ms/step - loss: 0.7776 - acc: 0.7500
Epoch 2/3
32/32 [==============================] - 2s 74ms/step - loss: 1.0575e-04 - acc: 1.0000
Epoch 3/3
32/32 [==============================] - 2s 74ms/step - loss: 0.3989 - acc: 0.8750
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
            "clock": 2799.5916944444434,
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
            "total": 134941597696
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
    "samples_per_second": 13.546549661477052,
    "time": 2.3622251274064183,
    "time_epoch": 2.3622251274064183
}
