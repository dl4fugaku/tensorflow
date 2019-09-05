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
