/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

// Returns the output and mask tensors
std::tuple<Tensor&, Tensor&> native_dropout_out(
    RuntimeContext& ctx,
    const Tensor& in,
    double p,
    optional<bool> train,
    Tensor& out,
    Tensor& mask) {

  // Define return value as tuple of the output values and bool mask of dropped out
  std::tuple<Tensor&, Tensor&> ret_val(out, mask);

  // Check that p is in [0, 1]
  ET_KERNEL_CHECK(ctx, p >= 0 && p <= 1, InvalidArgument, ret_val);

  // Rescale non-dropped out neurons for equal summated activation to next layer
  const double scale = 1.0 / (1.0 - p);

  // Define random boolean generator
  // Maybe a better way to ensure you get exact dropout percent but is efficient?
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution d(1.0 - p);

  // Resize output and mask tensors
  ET_KERNEL_CHECK(ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, ret_val);
  ET_KERNEL_CHECK(ctx, resize_tensor(mask, in.sizes()) == Error::Ok, InvalidArgument, ret_val);

  const auto& sizes = in.sizes();
  const int64_t batch_size = sizes[0];
  const int64_t channels = sizes[1];
  const int64_t height = sizes[2];
  const int64_t width = sizes[3];
  const int64_t channel_size = height * width;
  const int64_t in_dim = in.dim();

  // Operate on input, output, and mask data to gather and set values for return output
  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "native_dropout_out", CTYPE, [&]() {
    // the implementation at runtime/core/portable_type/tensor.h exposes
    // both constant and mutable data pointers. The CTYPE is acquired from the wrapper function above
    // and used to match the data type being used dynamically
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    bool* mask_data = mask.mutable_data_ptr<bool>();

    // If it is 2d image shaped, so 4 dimensions, and last 2 (height and width)
    // are same size, then do dropout2d
    if (in_dim == 4 && height == width) {
      // Dropout2d
      // std::cout << "using 2d" << std::endl;
      // For batches
      for (int64_t b = 0; b < batch_size; ++b) {
        // For channels
        for (int64_t c = 0; c < channels; ++c) {
          // Randomly determine if will keep or not
          bool keep = d(gen);
          // Define scaling, either 0 or the scale depending on keep
          CTYPE channel_scale = keep ? static_cast<CTYPE>(scale) : 0;
          // For height
          for (int64_t h = 0; h < height; ++h) {
            // For width
            for (int64_t w = 0; w < width; ++w) {
              // Get row major index
              int64_t idx = ((b * channels + c) * height + h) * width + w;
              // Set mask data boolean
              mask_data[idx] = keep;
              // Update output neurons
              out_data[idx] = in_data[idx] * channel_scale;
            }
          }
        }
      }
    } else {
      // Dropout (Dropout1d?)
      // For each element in the input
      for (int64_t i = 0; i < in.numel(); ++i) {
        // Randomly determine if will keep or not
        bool keep = d(gen);
        // Set mask data boolean
        mask_data[i] = keep;
        // Update output neurons. Rescale if keep, or zero if drop out
        out_data[i] = keep ? in_data[i] * static_cast<CTYPE>(scale) : 0;
      }

    }
  });

  // Return the values
  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
