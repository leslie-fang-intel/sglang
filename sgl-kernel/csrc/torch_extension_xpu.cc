/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Python.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include "sgl_kernel_ops.h"

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API Tensor testll_kernel(Tensor act, Tensor weight);

}
} // namespace native
} // namespace at

at::Tensor sgl_test_sycl(at::Tensor act, at::Tensor weight) {
  return at::native::xpu::testll_kernel(act, weight);
}

TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */
  m.def(
      "sgl_test_sycl(Tensor a, Tensor b) -> Tensor");
  m.impl("sgl_test_sycl", at::kXPU, &sgl_test_sycl);
}

REGISTER_EXTENSION(common_ops)
