# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, SyclExtension

root = Path(__file__).parent.resolve()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"

# Option 1: since torch xpu ops didn't expose these helpers to submit, we hardcode the path here
# may copy it into the projection
include_dirs = [
    "/home/gta/miniforge3/envs/leslie/include/python3.10/",
    root / "include",
    root / "csrc",
    "/4T-720/leslie/inductor/pytorch/third_party/torch-xpu-ops/src/",
]

sources = [
    "csrc/torch_extension_xpu.cc",
    "csrc/elementwise/sgl_test_sycl.sycl",
]

cxx_flags = ["-O3"]
libraries = ["c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

debug_mode = False

xpucc_flags = [
    "-DNDEBUG" if not debug_mode else "-DDEBUG",
    "-O3" if not debug_mode else "-O0",
    "-t=0",
    "-std=c++17",
]

ext_modules = [
    SyclExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": xpucc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=True,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
