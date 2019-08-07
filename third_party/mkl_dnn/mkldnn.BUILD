exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party/mkl_dnn:build_defs.bzl",
    "if_mkl_open_source_only",
)
load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

# Create the file mkldnn_version.h with MKL-DNN version numbers.
# Currently, the version numbers are hard coded here. If MKL-DNN is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1) Automatically get the version numbers from CMakeLists.txt.

template_rule(
    name = "mkldnn_version_h",
    src = "include/mkldnn_version.h.in",
    out = "include/mkldnn_version.h",
    substitutions = {
        "@MKLDNN_VERSION_MAJOR@": "0",
        "@MKLDNN_VERSION_MINOR@": "18",
        "@MKLDNN_VERSION_PATCH@": "0",
        "@MKLDNN_VERSION_HASH@": "N/A",
    },
)

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/gemm/*.cpp",
        "src/cpu/gemm/*.hpp",
        "src/cpu/gemm/f32/*.cpp",
        "src/cpu/gemm/f32/*.hpp",
        "src/cpu/gemm/s8x8s32/*.cpp",
        "src/cpu/gemm/s8x8s32/*.hpp",
        "src/cpu/rnn/*.cpp",
        "src/cpu/rnn/*.hpp",
        "src/cpu/xbyak/*.h",
    ]) + [":mkldnn_version_h"],
    hdrs = glob(["include/*"]),
    # from mkldnn/src/cpu/gemm/gemm.cpp, so we need to remove -UUSE_CBLAS
    #/* USE_MKL      USE_CBLAS       effect
    #* -------      ---------       ------
    #* yes          yes             use Intel(R) MKL CBLAS
    #* yes          no              use jit
    #* no           yes             system-dependent CBLAS
    #* no           no              use jit
    #*/
    copts = [
        "-fexceptions",
        "-DUSE_MKL",
        "-DUSE_CBLAS",
    ] + if_mkl_open_source_only([
        "-UUSE_MKL",
        #"-UUSE_CBLAS",
	"-I /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/openblas-0.3.6-gyvjlofuwcw25ouj42jazev25svqizuv/include",
        "-fopenmp",
	"-DJIT_PROFILING_VTUNE",
	"-I /scr0/jens/tensorflow/jitperf/include",
    ]) + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",  # only works with gcc
        ],
        # TODO(ibiryukov): enable openmp with clang by including libomp as a
        # dependency.
        ":clang_linux_x86_64": [],
        "//conditions:default": [],
    }),
    linkopts = [
        "-fopenmp",
        "-L /scr0/jens/spack/opt/spack/linux-centos7-x86_64/gcc-9.1.0/openblas-0.3.6-gyvjlofuwcw25ouj42jazev25svqizuv/lib",
        "-lopenblas",
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [ "@jitperf//:jithdr", "@openblas//:lib", "@gfortran//:lib" ] + select({
        #"@org_tensorflow//tensorflow:linux_x86_64": [
        #    "@mkl_linux//:mkl_headers",
        #    "@mkl_linux//:mkl_libs_linux",
        #],
        #"@org_tensorflow//tensorflow:macos": [
        #    "@mkl_darwin//:mkl_headers",
        #    "@mkl_darwin//:mkl_libs_darwin",
        #],
        #"@org_tensorflow//tensorflow:windows": [
        #    "@mkl_windows//:mkl_headers",
        #    "@mkl_windows//:mkl_libs_windows",
        #],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "mkldnn_single_threaded",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/gemm/*.cpp",
        "src/cpu/gemm/*.hpp",
        "src/cpu/gemm/f32/*.cpp",
        "src/cpu/gemm/f32/*.hpp",
        "src/cpu/gemm/s8x8s32/*.cpp",
        "src/cpu/gemm/s8x8s32/*.hpp",
        "src/cpu/rnn/*.cpp",
        "src/cpu/rnn/*.hpp",
        "src/cpu/xbyak/*.h",
    ]) + [":mkldnn_version_h"],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-DMKLDNN_THR=MKLDNN_THR_SEQ",  # Disables threading.
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
)
