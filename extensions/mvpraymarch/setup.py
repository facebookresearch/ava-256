# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    import torch

    extensions_dir = os.path.dirname(os.path.dirname(__file__))
    setup(
        name="mvpraymarch",
        ext_modules=[
            CUDAExtension(
                "mvpraymarchlib",
                sources=["mvpraymarch.cpp", "mvpraymarch_kernel.cu", "bvh.cu"],
                include_dirs=[os.path.join(extensions_dir, "include")],
                extra_compile_args={
                    "nvcc": [
                        "-use_fast_math",
                        "-arch=sm_70",
                        "-std=c++17",
                        "-lineinfo",
                    ]
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
