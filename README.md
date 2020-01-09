```
cd $HOME
wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-dist.zip
mkdir bazel-0.25.2; cd bazel-0.25.2
unzip ../bazel-0.25.2-dist.zip
sed -i -e 's/fcntl(fd, F_OFD_SETLK/fcntl(fd, F_SETLK/' ./src/main/cpp/blaze_util_posix.cc
PROTOC=pwd/bin/protoc EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" GRPC_JAVA_PLUGIN=pwd/grpc-java bash ./compile.sh
export PATH=$PATH:`pwd`/../bazel-0.25.2/output
```

```
. $HOME/spack/share/spack/setup-env.sh
spack install hdf5 hl=True
spack load hdf5
spack load openmpi
```

```
cd $HOME
git clone https://github.com/dl4fugaku/tensorflow.git
cd tensorflow
#git checkout -b fugaku-v2-a64fx v2.0.0
git checkout fugaku-v2-a64fx
python3.6 -m pip install --user --upgrade numpy grpcio tensorboard absl-py keras-applications gast astor termcolor opt-einsum google-pasta wrapt wheel six protobuf numpy keras-preprocessing keras_preprocessing tensorflow-estimator
rm -rf $HOME/.cache/bazel
rm -rf $HOME/.cache/log
TEST_TMPDIR=../.cache/bazel ./configure
TEST_TMPDIR=$HOME/.cache/bazel CC=gcc CXX=g++ ~/bazel-0.25.2/output/bazel --output_base=$HOME/.cache/log build -s --config=numa --config=v2 --config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl --copt=-march=native --copt=-O3 --copt=-finline-functions --copt=-findirect-inlining --cxxopt=-march=native --cxxopt=-O3 --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package --use_action_cache --verbose_failures --repository_cache=$HOME/.cache/bazel --disk_cache=$HOME/.cache/bazel --local_ram_resources=$((16*1024))
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg/
python3.6 -m pip install --user --upgrade ./tmp/tensorflow_pkg/tensorflow-2.0.0-cp36-cp36m-linux_aarch64.whl
```

```
cd $HOME
git clone --recursive https://github.com/undertherain/benchmarker.git
python3.6 -m pip install --user --upgrade 'system_query[cpu,hdd,ram,swap]'
cd benchmarker
python3.6 -m benchmarker --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4
```

