is this a guide?
================
DEF NOT!!!

patch spack packages
====================
```
diff --git a/var/spack/repos/builtin/packages/cubelib/package.py b/var/spack/repos/builtin/packages/cubelib/package.py
index 666a234..ea4f769 100644
--- a/var/spack/repos/builtin/packages/cubelib/package.py
+++ b/var/spack/repos/builtin/packages/cubelib/package.py
@@ -12,6 +12,7 @@ class Cubelib(AutotoolsPackage):
     homepage = "http://www.scalasca.org/software/cube-4.x/download.html"
     url = "http://apps.fz-juelich.de/scalasca/releases/cube/4.4/dist/cubelib-4.4.tar.gz"
 
+    version('4.4.4', 'adb8216ee3b7701383884417374e7ff946edb30e56640307c65465187dca7512')
     version('4.4.3', 'bcd4fa81a5ba37194e590a5d7c3e6c44b448f5e156a175837b77c21206847a8d')
     version('4.4.2', '843335c7d238493f1b4cb8e07555ccfe99a3fa521bf162e9d8eaa6733aa1f949')
     version('4.4',   'c903f3c44d3228ebefd00c831966988e')
diff --git a/var/spack/repos/builtin/packages/cubew/package.py b/var/spack/repos/builtin/packages/cubew/package.py
index aeb911c..e4aeb90 100644
--- a/var/spack/repos/builtin/packages/cubew/package.py
+++ b/var/spack/repos/builtin/packages/cubew/package.py
@@ -12,6 +12,7 @@ class Cubew(AutotoolsPackage):
     homepage = "http://www.scalasca.org/software/cube-4.x/download.html"
     url = "http://apps.fz-juelich.de/scalasca/releases/cube/4.4/dist/cubew-4.4.tar.gz"
 
+    version('4.4.3', '93fff6cc1e8b0780f0171ef5302a2e1a257f99b6383fbfc1b9b82f925ceff501')
     version('4.4.2', '31a71e9a05e6523de2b86b4026821bbb75fb411eb5b18ae38b27c1f44158014a')
     version('4.4.1', 'c09e3f5a3533ebedee2cc7dfaacd7bac4680c14c3fa540669466583a23f04b67')
     version('4.4',   'e9beb140719c2ad3d971e1efb99e0916')
diff --git a/var/spack/repos/builtin/packages/opari2/package.py b/var/spack/repos/builtin/packages/opari2/package.py
index 65ad8c8..aed1703 100644
--- a/var/spack/repos/builtin/packages/opari2/package.py
+++ b/var/spack/repos/builtin/packages/opari2/package.py
@@ -20,6 +20,7 @@ class Opari2(AutotoolsPackage):
     homepage = "http://www.vi-hps.org/projects/score-p"
     url      = "https://www.vi-hps.org/cms/upload/packages/opari2/opari2-2.0.4.tar.gz"
 
+    version('2.0.5', '9034dd7596ac2176401090fd5ced45d0ab9a9404444ff767f093ccce68114ef5')
     version('2.0.4', 'f69e324792f66780b473daf2b3c81f58ee8188adc72b6fe0dacf43d4c1a0a131')
     version('2.0.3', 'f34674718ffdb098a48732a1eb9c1aa2')
     version('2.0.1', '74af78f1f27b8caaa4271e0b97fb0fba')
diff --git a/var/spack/repos/builtin/packages/otf2/package.py b/var/spack/repos/builtin/packages/otf2/package.py
index 71571ba..bf5f6eb 100644
--- a/var/spack/repos/builtin/packages/otf2/package.py
+++ b/var/spack/repos/builtin/packages/otf2/package.py
@@ -15,6 +15,7 @@ class Otf2(AutotoolsPackage):
     homepage = "http://www.vi-hps.org/projects/score-p"
     url      = "https://www.vi-hps.org/cms/upload/packages/otf2/otf2-2.1.1.tar.gz"
 
+    version('2.2',   'd0519af93839dc778eddca2ce1447b1ee23002c41e60beac41ea7fe43117172d')
     version('2.1.1', 'e51ad0d8ca374d25f47426746ca629e7')
     version('2.1',   'e2994e53d9b7c2cbd0c4f564d638751e')
     version('2.0',   '5b546188b25bc1c4e285e06dddf75dfc')
diff --git a/var/spack/repos/builtin/packages/py-numpy/package.py b/var/spack/repos/builtin/packages/py-numpy/package.py
index e438598..48f154f 100644
--- a/var/spack/repos/builtin/packages/py-numpy/package.py
+++ b/var/spack/repos/builtin/packages/py-numpy/package.py
@@ -26,6 +26,7 @@ class PyNumpy(PythonPackage):
         'numpy.distutils.command', 'numpy.distutils.fcompiler'
     ]
 
+    version('1.17.2', sha256='73615d3edc84dd7c4aeb212fa3748fb83217e00d201875a47327f55363cef2df')
     version('1.16.4', sha256='7242be12a58fec245ee9734e625964b97cf7e3f2f7d016603f9e56660ce479c7')
     version('1.16.3', sha256='78a6f89da87eeb48014ec652a65c4ffde370c036d780a995edaeb121d3625621')
     version('1.16.2', sha256='6c692e3879dde0b67a9dc78f9bfb6f61c666b4562fd8619632d7043fb5b691b0')
diff --git a/var/spack/repos/builtin/packages/scorep/package.py b/var/spack/repos/builtin/packages/scorep/package.py
index 83be646..17a5640 100644
--- a/var/spack/repos/builtin/packages/scorep/package.py
+++ b/var/spack/repos/builtin/packages/scorep/package.py
@@ -15,6 +15,7 @@ class Scorep(AutotoolsPackage):
     homepage = "http://www.vi-hps.org/projects/score-p"
     url      = "https://www.vi-hps.org/cms/upload/packages/scorep/scorep-4.1.tar.gz"
 
+    version('6.0',   '5dc1023eb766ba5407f0b5e0845ec786e0021f1da757da737db1fb71fc4236b8')
     version('5.0',   '0651614eacfc92ffbe5264a3efebd0803527ae6e8b11f7df99a56a02c37633e1')
     version('4.1',   '7bb6c1eecdd699b4a3207caf202866778ee01f15ff39a9ec198fcd872578fe63')
     version('4.0',   'f04478e0407d67eeb8c49c3c51d91e12')
@@ -25,19 +26,26 @@ class Scorep(AutotoolsPackage):
     version('1.3',   '9db6f957b7f51fa01377a9537867a55c')
 
     patch('gcc7.patch', when='@:3')
+    patch('TFfix14.patch', when='@6:')
+    patch('TFfix16.patch', when='@6:')
 
     variant('mpi', default=True, description="Enable MPI support")
     variant('papi', default=True, description="Enable PAPI")
     variant('pdt', default=False, description="Enable PDT")
     variant('shmem', default=False, description='Enable shmem tracing')
+    variant('libunwind', default=False, description="Enable sampling via libunwind")
 
     # Dependencies for SCORE-P are quite tight. See the homepage for more
     # information. Starting with scorep 4.0 / cube 4.4, Score-P only depends on
     # two components of cube -- cubew and cubelib.
 
+    depends_on('otf2@2.2:', when='@6:')
+    depends_on('opari2@2.0.5:', when='@6:')
+    depends_on('cubew@4.4.3:', when='@6:')
+    depends_on('cubelib@4.4.4:', when='@6:')
     # SCOREP 4 and 5
-    depends_on('otf2@2.1:', when='@4:')
-    depends_on('opari2@2.0:', when='@4:')
+    depends_on('otf2@2.1:', when='@4:5')
+    depends_on('opari2@2.0:', when='@4:5')
     depends_on('cubew@4.4:', when='@4:')
     depends_on('cubelib@4.4:', when='@4:')
     # SCOREP 3
@@ -60,6 +68,7 @@ class Scorep(AutotoolsPackage):
     depends_on('mpi', when="+mpi")
     depends_on('papi', when="+papi")
     depends_on('pdt', when="+pdt")
+    depends_on('libunwind', when="+libunwind")
 
     # Score-P requires a case-sensitive file system, and therefore
     # does not work on macOS
@@ -72,7 +81,7 @@ class Scorep(AutotoolsPackage):
         config_args = [
             "--with-otf2=%s" % spec['otf2'].prefix.bin,
             "--with-opari2=%s" % spec['opari2'].prefix.bin,
-            "--enable-shared"]
+            "--enable-shared", "--disable-online-access"]
 
         cname = spec.compiler.name
         config_args.append('--with-nocross-compiler-suite={0}'.format(cname))
@@ -92,6 +101,9 @@ class Scorep(AutotoolsPackage):
         if "+pdt" in spec:
             config_args.append("--with-pdt=%s" % spec['pdt'].prefix.bin)
 
+        if "+libunwind" in spec:
+            config_args.append("--with-libunwind=%s" % spec['libunwind'].prefix)
+
         config_args += self.with_or_without('shmem')
         config_args += self.with_or_without('mpi')
 
diff --git a/var/spack/repos/builtin/packages/vampirtrace/package.py b/var/spack/repos/builtin/packages/vampirtrace/package.py
index 4b142f8..aecc66b 100644
--- a/var/spack/repos/builtin/packages/vampirtrace/package.py
+++ b/var/spack/repos/builtin/packages/vampirtrace/package.py
@@ -51,7 +51,11 @@ class Vampirtrace(AutotoolsPackage):
             '--with-wrapper-cxx-compiler={0}'.format(compiler.cxx),
             '--with-wrapper-cxx-cpp={0} -E'.format(compiler.cxx),
             '--with-wrapper-fc-compiler={0}'.format(compiler.fc),
-            '--with-wrapper-fc-cpp={0} -E'.format(compiler.fc)
+            '--with-wrapper-fc-cpp={0} -E'.format(compiler.fc),
+            '-enable-compinst=gnu',
+            '-disable-cudartwrap',
+            '-disable-java',
+            '-disable-cupti'
         ]
 
         if '+mpi' in spec:
```

need patches for scorep internals
=================================
```
⇒  cat var/spack/repos/builtin/packages/scorep/TFfix14.patch
diff -Nur scorep-5.0.old/src/tools/instrumenter/scorep_instrumenter_cmd_line.cpp scorep-5.0.new/src/tools/instrumenter/scorep_instrumenter_cmd_line.cpp
--- scorep-5.0.old/src/tools/instrumenter/scorep_instrumenter_cmd_line.cpp      2019-03-23 02:41:48.951808977 +0900
+++ scorep-5.0.new/src/tools/instrumenter/scorep_instrumenter_cmd_line.cpp      2019-09-26 16:17:02.276421925 +0900
@@ -600,8 +600,9 @@
         }
         add_library( current );
     }
-    else if ( ( current[ 0 ] != '-' ) &&
-              ( is_source_file( current ) || is_object_file( current ) ) )
+    else if ( ( current.find(".params") != std::string::npos ) ||
+              (( current[ 0 ] != '-' ) &&
+              ( is_source_file( current ) || is_object_file( current ) )) )
     {
         add_input_file( current );
         return scorep_parse_mode_command;
```
```
⇒  cat var/spack/repos/builtin/packages/scorep/TFfix16.patch
diff -Nur scorep-6.0.old/src/measurement/tracing/SCOREP_Tracing.c scorep-6.0.new/src/measurement/tracing/SCOREP_Tracing.c
--- scorep-6.0.old/src/measurement/tracing/SCOREP_Tracing.c     2019-07-31 16:06:13.492697187 +0900
+++ scorep-6.0.new/src/measurement/tracing/SCOREP_Tracing.c     2019-10-15 21:57:54.340880932 +0900
@@ -486,7 +486,7 @@
 
     /* OTF2 does the broadcast for us */
     ret = OTF2_Archive_SetDefChunkSize( scorep_otf2_archive,
-                                        definition_chunk_size );
+                                        definition_chunk_size*4 );
     if ( OTF2_SUCCESS != ret && SCOREP_Status_GetRank() == 0 )
     {
         UTILS_FATAL( "Could not set OTF2 definition chunks size to %" PRIu64 ": %s",
```

install more spacks
===================
spack install openmpi@3.1.4 vt=False ^autoconf@2.69 ^automake@1.16.1 ^libtool@2.4.6 ^m4@1.4.18 ^perl@5.26.2
spack install py-mpi4py ^openmpi@3.1.4/5g7q2fs ^py-setuptools@41.0.1/4f5j3zz ^python@3.7.3/civ2iiq
spack install libunwind
spack install scorep libunwind=True ^openmpi@3.1.4/5g7q2fs

spack load py-mpi4py; spack load scorep

build numpy against atlas (to avoid issues with scorep-enabled openblas)
========================================================================
spack install atlas threads=pthreads tune_cpu=18
spack install py-numpy@1.17.2 ^atlas@3.10.2/2yvevgj ^py-setuptools@41.0.1/4f5j3zz ^python@3.7.3/civ2iiq

need scorep pything support
===========================
git clone https://github.com/score-p/scorep_binding_python.git
cd scorep_binding_python
python3.7 -m pip install --upgrade .

build a scorep-enabled openblas version
=======================================
first install normal threaded version: spack install openblas threads=pthreads
then overwrite it from source:
```
wget https://github.com/xianyi/OpenBLAS/archive/v0.3.7.tar.gz
tar xzf v0.3.7.tar.gz
cd OpenBLAS-0.3.7/
make BINARY=64 CC=scorep-gcc FC=scorep-gfortran USE_OPENMP=0 USE_THREAD=1 NO_LAPACK=1 COMMON_OPT=-O3 COMMON_PROF="" USE_TLS=1 GEMM_MULTITHREAD_THRESHOLD=1 NUM_THREADS=36 libs shared
make install PREFIX=/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/openblas-0.3.6-f67sdjaj5bqkvesbepb5lhbb6zoa4rvn
```

patch benchmarker
=================
```
diff --git a/benchmarker/modules/do_tensorflow.py b/benchmarker/modules/do_tensorflow.py
index 6431623..999e820 100644
--- a/benchmarker/modules/do_tensorflow.py
+++ b/benchmarker/modules/do_tensorflow.py
@@ -2,12 +2,19 @@
 """TensorFlow support.
 """
 
+import scorep
 import os
 from timeit import default_timer as timer
 from .i_neural_net import INeuralNet
 from benchmarker.util.data import to_categorical
 import tensorflow as tf
 
+tf_inter = os.getenv('TF_INTER')
+tf_intra = os.getenv('TF_INTRA')
+if not tf_inter: tf_inter = '2'
+if not tf_intra: tf_intra = '16'
+tf.config.threading.set_inter_op_parallelism_threads(int(tf_inter))
+tf.config.threading.set_intra_op_parallelism_threads(int(tf_intra))
 
 class DoTensorflow(INeuralNet):
     """docstring for ClassName"""
@@ -46,11 +53,17 @@ class DoTensorflow(INeuralNet):
         if self.params["mode"] != "training":
             raise NotADirectoryError("only training is implemented for TF")
         print("preheat")
+        scorep.user.region_begin("preheat")
         model.fit(x_train, y_train, batch_size=self.params["batch_size"], epochs=1)
-        nb_epoch = 3
+        scorep.user.region_end("preheat")
+#        nb_epoch = 3
+        nb_epoch = 1
         print("train")
         start = timer()
+        scorep.user.rewind_end('ignore_init_and_preheat', True)
+        scorep.user.region_begin("train")
         model.fit(x_train, y_train, batch_size=self.params["batch_size"], epochs=nb_epoch, verbose=1)
+        scorep.user.region_end("train")
         end = timer()
         self.params["time"] = (end - start) / nb_epoch
         version_backend = tf.__version__
@@ -61,5 +74,6 @@ class DoTensorflow(INeuralNet):
 
 
 def run(params):
+    scorep.user.enable_recording()
     backend_tf = DoTensorflow(params)
     return backend_tf.run()
```
and need:
```
⇒  cat bechmarker-wrapper.py 
import scorep
scorep.user.rewind_begin('ignore_init_and_preheat')
scorep.user.disable_recording()

import argparse
from benchmarker import benchmarker

def main():
    parser = argparse.ArgumentParser(description='Benchmark me up, Scotty!')
    parser.add_argument("--framework")
    parser.add_argument("--problem")
    parser.add_argument('--path_out', type=str, default="./logs")
    parser.add_argument('--gpus', default="")
    parser.add_argument('--problem_size', default=None)
    parser.add_argument('--batch_size', default=None)
    # TODO: let submodules define their own extra parameters
    parser.add_argument('--mode', default=None)
    parser.add_argument('--misc')
    args = parser.parse_args()
    benchmarker.run(args)


if __name__ == "__main__":
    main()
```

patch tensorflow
================
```
diff --git a/WORKSPACE b/WORKSPACE
index 74ea14d..3167497 100644
--- a/WORKSPACE
+++ b/WORKSPACE
@@ -143,3 +143,36 @@ http_archive(
         "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
     ],
 )
+
+new_local_repository(
+    name = "openblas",
+#    path = "/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/openblas-0.3.6-gyvjlofuwcw25ouj42jazev25svqizuv",
+    path = "/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/openblas-0.3.6-f67sdjaj5bqkvesbepb5lhbb6zoa4rvn",
+    build_file_content = """
+package(default_visibility = ["//visibility:public"])
+cc_library(
+    name = "lib",
+#    srcs = glob(["lib/libopenblas.a"]),
+    srcs = glob(["lib/libopenblas.so"]),
+#    hdrs = glob(["include/*.h"]),
+)
+cc_library(
+    name = "incl",
+    srcs = glob(["include/*.h"]),
+    includes = ["include"],
+)
+"""
+)
+
+new_local_repository(
+    name = "gfortran",
+    path = "/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/gcc-9.1.0-qn3nra4nquabplipzj3k4v6aff2jd4tr/lib64",
+    build_file_content = """
+package(default_visibility = ["//visibility:public"])
+cc_library(
+    name = "lib",
+    srcs = glob(["libgfortran.so", "libquadmath.so", "libgomp.so"]),
+)
+"""
+)
+
```
```
diff --git a/tensorflow/tensorflow.bzl b/tensorflow/tensorflow.bzl
index 97b10bf..e2f24c8 100644
--- a/tensorflow/tensorflow.bzl
+++ b/tensorflow/tensorflow.bzl
@@ -292,6 +292,7 @@ def tf_copts(android_optimization_level_override = "-O2", is_external = False):
         if_tensorrt(["-DGOOGLE_TENSORRT=1"]) +
         if_mkl(["-DINTEL_MKL=1", "-DEIGEN_USE_VML"]) +
         if_mkl_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) +
+        if_mkl_v1_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) +
         if_mkl_v1_open_source_only(["-DENABLE_MKLDNN_V1"]) +
         if_enable_mkl(["-DENABLE_MKL"]) +
         if_ngraph(["-DINTEL_NGRAPH=1"]) +
@@ -313,7 +314,7 @@ def tf_copts(android_optimization_level_override = "-O2", is_external = False):
     )
 
 def tf_openmp_copts():
-    return if_mkl_lnx_x64(["-fopenmp"])
+    return if_mkl_lnx_x64(["-fopenmp"]) #[]) #["-fopenmp"])
 
 def tfe_xla_copts():
     return select({
@@ -378,7 +379,7 @@ def _rpath_linkopts(name):
     # ops) are picked up as long as they are in either the same or a parent
     # directory in the tensorflow/ tree.
     levels_to_root = native.package_name().count("/") + name.count("/")
-    return select({
+    return ["-lopenblas"] + select({
         clean_dep("//tensorflow:macos"): [
             "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
         ],
@@ -1363,7 +1364,7 @@ def tf_gpu_library(deps = None, cuda_deps = None, copts = tf_copts(), **kwargs):
         ]) + if_rocm_is_configured(cuda_deps + [
             "@local_config_rocm//rocm:rocm_headers",
         ]),
-        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"]) + if_mkl(["-DINTEL_MKL=1"]) + if_mkl_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) + if_enable_mkl(["-DENABLE_MKL"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
+        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"]) + if_mkl(["-DINTEL_MKL=1"]) + if_mkl_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) + if_mkl_v1_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) + if_enable_mkl(["-DENABLE_MKL"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
         **kwargs
     )
 
```
```
diff --git a/third_party/eigen.BUILD b/third_party/eigen.BUILD
index 6b585e7..8dfbcd3 100644
--- a/third_party/eigen.BUILD
+++ b/third_party/eigen.BUILD
@@ -67,6 +67,9 @@ cc_library(
         "EIGEN_MAX_ALIGN_BYTES=64",
         "EIGEN_HAS_TYPE_TRAITS=0",
         "EIGEN_NO_DEBUG",
+        "EIGEN_USE_BLAS",
+        "EIGEN_HAS_CXX11",
+#        "EIGEN_DONT_PARALLELIZE",
     ],
     includes = ["."],
     visibility = ["//visibility:public"],
```
and need some extra files
a) third_party/toolchains/cpus/arm/cc_config.bzl.tpl
b)
```
⇒  cat gcc
#!/bin/bash

/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/scorep-6.0-tk2cm6cbwyh32p3xwimn2hrwuzilsllt/bin/scorep --thread=pthread --nopomp --noopenmp --instrument-filter=/scr0/jens/tensorflow/eigen.filter /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/gcc-9.1.0-qn3nra4nquabplipzj3k4v6aff2jd4tr/bin/gcc $@
```
```
⇒  cat g++
#!/bin/bash

/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/scorep-6.0-tk2cm6cbwyh32p3xwimn2hrwuzilsllt/bin/scorep --thread=pthread --nopomp --noopenmp --instrument-filter=/scr0/jens/tensorflow/eigen.filter /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/gcc-9.1.0-qn3nra4nquabplipzj3k4v6aff2jd4tr/bin/g++ $@
```
```
⇒  cat eigen.filter
SCOREP_FILE_NAMES_BEGIN
        INCLUDE */Eigen/*
SCOREP_FILE_NAMES_END
SCOREP_REGION_NAMES_BEGIN
        INCLUDE Eigen*
        EXCLUDE Eigen::internal::TensorContractionSubMapper*
                Eigen::internal::TensorContractionInputMapper*
SCOREP_REGION_NAMES_END
```

building TF itself via faked cross compile to avoid scorep issues
=================================================================
might be good to build against the other openblas (if using share libs), to
avoid stupid stuff with scorep during builds and then softlink or ldpreload later
```
PYPATH=`which python`
PY2PATH=`dirname ${PYPATH}`
PY2PATH=`dirname ${PY2PATH}`
BLAS_INSTALL_PATH=`echo ${TFBASE}/spack/opt/spack/linux-*/gcc-9.1.0/openblas-0.3.6-f*`
BLAS_LIB_A=libopenblas.so
# need gcc wrapper to add eigen filter in root dir, but dont put into PATH, its handled by the arm toolchain
TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --output_base=${TFBASE}/log build --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --crosstool_top=@local_config_arm_compiler//:toolchain \
        -s --define=tensorflow_mkldnn_contraction_kernel=0 --config=numa --config=v2 \
        --config=noaws --config=nohdfs --config=noignite --config=nokafka --config=nonccl \
        --copt=-march=native --copt=-O3 --copt=-finline-functions --copt=-findirect-inlining \
        --cxxopt=-march=native --cxxopt=-O3 --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
	--copt=-I${BLAS_INSTALL_PATH}/include --linkopt=-L${BLAS_INSTALL_PATH}/lib --linkopt=-l:${BLAS_LIB_A} \
        //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow.so //tensorflow:libtensorflow_cc.so \
        --use_action_cache --verbose_failures --repository_cache=${TFBASE}/.cache/bazel --disk_cache=${TFBASE}/.cache/bazel --local_cpu_resources=36
```

runing the freaking mess
========================
SCOREP_FILTERING_FILE=/scr0/jens/benchmarker/eigen.runfilter SCOREP_TOTAL_MEMORY=3900M SCOREP_ENABLE_PROFILING=true SCOREP_ENABLE_TRACING=false python3.7 -m scorep --thread=pthread --nopython ./bechmarker-wrapper.py --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4
OPENBLAS_NUM_THREADS=18 SCOREP_FILTERING_FILE=/scr0/jens/benchmarker/eigen.runfilter SCOREP_ENABLE_UNWINDING=true SCOREP_TOTAL_MEMORY=3900M SCOREP_ENABLE_PROFILING=false SCOREP_ENABLE_TRACING=true python3.7 -m scorep --nopython --noinstrumenter --thread=pthread ./bechmarker-wrapper.py --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4
OPENBLAS_NUM_THREADS=18 SCOREP_ENABLE_UNWINDING=true SCOREP_METRIC_PAPI="PAPI_L2_TCA,PAPI_L2_TCM,PAPI_L3_TCM" SCOREP_TOTAL_MEMORY=3900M SCOREP_ENABLE_PROFILING=false SCOREP_ENABLE_TRACING=true python3.7 -m scorep --nopython --noinstrumenter --thread=pthread ./bechmarker-wrapper.py --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4

optional runtime filter if needed
=================================
```
⇒  cat eigen.runfilter
SCOREP_REGION_NAMES_BEGIN
        EXCLUDE Eigen::internal::TensorContractionSubMapper*
        EXCLUDE Eigen::internal::TensorContractionInputMapper*
SCOREP_REGION_NAMES_END
```

sgemm wrapper is also fun
=========================
```
⇒  cat sgemm-wrapper.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "cblas.h"


void (*real_cblas_sgemm)(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST     blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float * C, OPENBLAS_CONST blasint ldc);

void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST     blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float * C, OPENBLAS_CONST blasint ldc)
{
        printf("cblas_sgemm RowM, NoTrA, NoTrB, m, n, k lda, ldb, ldc, al, be = (%u, %u, %u, %d, %d, %d, %d, %d, %d, %e, %e)\n", Order, TransA, TransB, M, N, K, lda, ldb, ldc, alpha, beta);
        //real_cblas_sgemm = dlsym(RTLD_NEXT, "cblas_sgemm");
        void* handle = dlopen("/scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/openblas-0.3.6-gyvjlofuwcw25ouj42jazev25svqizuv/lib/libopenblas.so.0", RTLD_LAZY);
        real_cblas_sgemm = dlsym(handle, "cblas_sgemm");
        assert(real_cblas_sgemm);
        real_cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
}
```
gcc -Wall -fPIC -shared -o myfopen.so sgemm-wrapper.c -ldl
LD_PRELOAD=/scr0/jens/benchmarker/myfopen.so python3.7 -m scorep --nopython --thread=pthread ./bechmarker-wrapper.py --mode=training --framework=tensorflow --problem=resnet50 --problem_size=32 --batch_size=4


getting nvidia support into tf
==============================
echo "${PYPATH}\n${PY2PATH}/lib/python3.7/site-packages\ny\nn\nn\ny\nn\n7.0\nn\n\nn\n\n\n" \
 | CUDA_TOOLKIT_PATH=/usr/local/cuda,/usr/local/cuda/bin,/usr/local/cuda/targets/aarch64-linux/include,/usr/local/cuda/targets/aarch64-linux/lib TF_CUDA_VERSION=10.2 TEST_TMPDIR=../.cache/bazel ./configure
TEST_TMPDIR=../.cache/bazel CC=gcc CXX=g++ bazel --output_base=${TFBASE}/log build \
        -s --config=numa --config=v2 --config=cuda \
        --config=noaws --config=nohdfs --config=noignite --config=nokafka \
        --copt=-march=native --copt=-O3 --copt=-finline-functions --copt=-findirect-inlining \
        --cxxopt=-march=native --cxxopt=-O3 --cxxopt=-finline-functions --cxxopt=-findirect-inlining --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
        //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow.so //tensorflow:libtensorflow_cc.so \
        --use_action_cache --verbose_failures --repository_cache=${TFBASE}/.cache/bazel --disk_cache=${TFBASE}/.cache/bazel --local_cpu_resources=36
python3.7 -m benchmarker --mode=training --framework=tensorflow --problem=resnet50 --problem_size=512 --batch_size=32

