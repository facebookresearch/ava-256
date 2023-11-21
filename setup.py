import os
import re
import sys
import socket
import glob
import subprocess
from setuptools import setup

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

# Create a custom subclass that doesn't use Ninja, since Ninja doesn't support
# building multiple extension modules in parallel.
class BuildExtension2(BuildExtension):
    def __init__(self, *args, **kwargs):
        kwargs["use_ninja"] = False
        super().__init__(*args, **kwargs)

# From pytorch/torch/utils/cpp_extension.py
# https://github.com/pytorch/pytorch/blob/385165ec674b764eb42ffe396f98fadd08a513eb/torch/utils/cpp_extension.py#L24
IS_WINDOWS = sys.platform == "win32"

def _find_cuda_home():
    """Finds the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        try:
            which = "where" if IS_WINDOWS else "which"
            with open(os.devnull, "w") as devnull:
                nvcc = (
                    subprocess.check_output([which, "nvcc"], stderr=devnull).decode().rstrip("\r\n")
                )
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home is None:
        print("CUDA not found, set CUDA_HOME environment variable")
        exit(1)
    return cuda_home

def _get_cuda_info():
    cuda_home = _find_cuda_home()
    print("CUDA Home: (set env var CUDA_HOME to change): ", cuda_home)

    nvcc_bin = os.path.join(cuda_home, "bin", "nvcc")
    bin2c_bin = os.path.join(cuda_home, "bin", "bin2c")
    if IS_WINDOWS:
        nvcc_bin += ".exe"
        bin2c_bin += ".exe"

    cuda_ver_str = subprocess.check_output([nvcc_bin, "--version"])
    cuda_ver = int(re.search(r"release ([0-9]+)(\.[0-9]+)?", cuda_ver_str.decode("utf8")).group(1))

    centos_gcc8 = "/opt/rh/devtoolset-8/root/usr/bin/gcc"
    if os.path.exists(centos_gcc8):
        print(f"Setting GCC to: {centos_gcc8}")
        os.environ["CC"] = centos_gcc8
        os.environ["CXX"] = centos_gcc8.replace("gcc", "g++")

    cxx_args = ["-std=c++14"]
    if IS_WINDOWS:
        cxx_args.append("/O2")
    else:
        cxx_args.append("-O3")
    nvcc_args = ["-O3"]
    nvcc_ccbin = None

    if IS_WINDOWS:
        nvcc_args.extend(
            [
                "-gencode=arch=compute_60,code=compute_60",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
            ]
        )
        cxx_args.extend(
            [
                "-DHAVE_ARCH60=1",
                "-DHAVE_ARCH61=1",
                "-DHAVE_ARCH70=1",
                "-DHAVE_ARCH75=1",
            ]
        )
        compute_caps = ["60", "61", "70", "75"]

        if cuda_ver >= 11:
            nvcc_args.extend(
                [
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                ]
            )
            cxx_args.extend(["-DHAVE_ARCH80=1", "-DHAVE_ARCH86=1"])
            compute_caps.extend(["80", "86"])
    else:
        pit_clusters = {"pit101", "pit103"}
        hostname = socket.gethostname().lower()
        is_pit_cluster = any(cname in hostname for cname in pit_clusters)

        is_rsc_cluster = "rsc" in hostname


        if is_pit_cluster or is_rsc_cluster:
            print("Pittsburgh DGX cluster detected.")
            nvcc_args.extend(
                [
                    "-gencode=arch=compute_60,code=compute_60",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                ]
            )
            cxx_args.extend(["-DHAVE_ARCH60=1", "-DHAVE_ARCH61=1", "-DHAVE_ARCH70=1"])
            compute_caps = ["60", "61", "70"]
        else:
            print(f"Unknown host: {hostname}")
            username = os.getenv("USER")
            homedir = os.environ.get("HOME", os.path.join("/home", username))
            nvcc_ccbin = "-ccbin={homedir}/cuda_compilers".format(homedir=homedir)

            nvcc_args.extend(
                [
                    "-gencode=arch=compute_60,code=compute_60",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                ]
            )
            cxx_args.append("-DHAVE_ARCH60=1")
            cxx_args.append("-DHAVE_ARCH61=1")
            cxx_args.append("-DHAVE_ARCH70=1")
            cxx_args.append("-DHAVE_ARCH75=1")
            compute_caps = ["60", "61", "70", "75"]
            cxx_args.append("-I/usr/include/cuda")

    if "CC" in os.environ and nvcc_ccbin is None:
        nvcc_ccbin = "-ccbin={ccdir}".format(ccdir=os.path.dirname(os.environ.get("CC")))

    # PyTorch starting with 1.7 automatically includes -ccbin in the args if
    # you have CC set in your environment.
    if len(torch.__version__) > 3 and float(torch.__version__[:3]) < 1.7 and nvcc_ccbin is not None:
        nvcc_args.extend([nvcc_ccbin])

    return {
        "cuda_ver": cuda_ver,
        "cxx_args": cxx_args,
        "nvcc_args": nvcc_args,
        "nvcc_bin": nvcc_bin,
        "nvcc_ccbin": nvcc_ccbin,
        "bin2c_bin": bin2c_bin,
        "compute_caps": compute_caps,
    }

if __name__ == "__main__":
    centos_gcc8 = "/opt/rh/devtoolset-8/root/usr/bin/gcc"
    if os.path.exists(centos_gcc8):
        print(f"Setting GCC to: {centos_gcc8}")
        os.environ["CC"] = centos_gcc8
        os.environ["CXX"] = centos_gcc8.replace("gcc", "g++")

    cuda_info = _get_cuda_info()
    cxx_args = cuda_info["cxx_args"]
    nvcc_args = cuda_info["nvcc_args"]

    # Workaround for CUDAExtension bug where it fails to handle relative
    # include paths properly.
    common_incdir = os.path.join(os.getcwd(), "extensions", "common")
    mvpraymarch_incdir = os.path.join(os.getcwd(), "extensions", "mvpraymarch")

    setup(
        name="extensions",
        ext_modules=[
            CUDAExtension(
                "extensions.mvpraymarch.mvpraymarch_ext",
                sources=[
                    "extensions/mvpraymarch/mvpraymarch.cpp",
                    "extensions/mvpraymarch/mvpraymarch_kernel.cu",
                    "extensions/mvpraymarch/bvh.cu"
                ],
                include_dirs = [common_incdir],
                extra_compile_args={
                    "cxx":  cxx_args,
                    "nvcc": nvcc_args
                }
            ),
            CUDAExtension(
                "extensions.integprior.integprior_ext",
                sources=[
                    "extensions/integprior/integprior.cpp",
                    "extensions/integprior/integprior_kernel.cu",
                ],
                include_dirs = [common_incdir, mvpraymarch_incdir],
                extra_compile_args={
                    "cxx":  cxx_args,
                    "nvcc": nvcc_args
                }
            ),
            CUDAExtension(
                "extensions.primintersection.primintersection_ext",
                sources=[
                    "extensions/primintersection/primintersection.cpp",
                    "extensions/primintersection/primintersection_kernel.cu",
                ],
                include_dirs = [common_incdir, mvpraymarch_incdir],
                extra_compile_args={
                    "cxx":  cxx_args,
                    "nvcc": nvcc_args
                }
            ),
            CUDAExtension(
                "extensions.computeraydirs.computeraydirs_ext",
                sources=[
                    "extensions/computeraydirs/computeraydirs.cpp",
                    "extensions/computeraydirs/computeraydirs_kernel.cu",
                ],
                include_dirs = [common_incdir],
                extra_compile_args={
                    "cxx":  cxx_args,
                    "nvcc": nvcc_args
                }
            ),
            CppExtension(
                "extensions.png_reader.png_reader_ext",
                sources=[
                    "extensions/png_reader/module.cpp",
                ],
                extra_link_args=["-flto"],
                include_dirs=['src/zip_reader'],
                extra_compile_args=cxx_args
            ),
        ],
        cmdclass={"build_ext": BuildExtension2},
        packages=[
            "extensions",
            "extensions.mvpraymarch",
            "extensions.primintersection",
            "extensions.computeraydirs",
            "extensions.png_reader",
        ],
    )
