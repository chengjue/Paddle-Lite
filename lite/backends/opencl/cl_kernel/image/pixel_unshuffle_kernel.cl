/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <cl_common.h>
__kernel void pixel_unshuffle(__read_only image2d_t input_image,
        __write_only image2d_t output_image,
        __private const int in_N,
        __private const int in_C,
        __private const int in_H,
        __private const int in_W,
        __private const int out_N,
        __private const int out_C,
        __private const int out_H,
        __private const int out_W,
        __private const int downscale_factor) {

    const int in_c4 = get_global_id(0);
    const int in_w = get_global_id(1);
    const int in_nh = get_global_id(2);

    int in_h = in_nh % in_H;
    int in_n = in_nh / in_H;

    CL_DTYPE4 res = (CL_DTYPE4)(0, 0, 0, 0);
    CL_DTYPE4 in;

    int in_c0 = in_c4 * 4 + 0;
    int out_c0 = in_c0 / (downscale_factor * downscale_factor);
    int offset0 = in_c0 % (downscale_factor * downscale_factor);
    int offset_h0 = offset0 / downscale_factor;
    int offset_w0 = offset0 % downscale_factor;

    int out_w0 = in_w * downscale_factor + offset_w0;
    int out_h0 = in_h * downscale_factor + offset_h0;
    int out_nh0 = in_n * out_H + out_h0;

    int2 out_pos0;
    out_pos0.x = out_w0 + (out_c0 / 4) * in_W;
    out_pos0.y = out_nh0;

    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos0);
    if (out_c0 % 4 == 0) {
        res.x = in.x;
    } else if (out_c0 % 4 == 1) {
        res.x = in.y;
    } else if (out_c0 % 4 == 2) {
        res.x = in.z;
    } else if (out_c0 % 4 == 3) {
        res.x = in.w;
    }

    int in_c1 = in_c4 * 4 + 1;
    int out_c1 = in_c1 / (downscale_factor * downscale_factor);
    int offset1 = in_c1 % (downscale_factor * downscale_factor);
    int offset_h1 = offset1 / downscale_factor;
    int offset_w1 = offset1 % downscale_factor;

    int out_w1 = in_w * downscale_factor + offset_w1;
    int out_h1 = in_h * downscale_factor + offset_h1;
    int out_nh1 = in_n * out_H + out_h1;

    int2 out_pos1;
    out_pos1.x = out_w1 + (out_c1 / 4) * in_W;
    out_pos1.y = out_nh1;

    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos1);
    if (out_c1 % 4 == 0) {
        res.y = in.x;
    } else if (out_c1 % 4 == 1) {
        res.y = in.y;
    } else if (out_c1 % 4 == 2) {
        res.y = in.z;
    } else if (out_c1 % 4 == 3) {
        res.y = in.w;
    }

    int in_c2 = in_c4 * 4 + 2;
    int out_c2 = in_c2 / (downscale_factor * downscale_factor);
    int offset2 = in_c2 % (downscale_factor * downscale_factor);
    int offset_h2 = offset2 / downscale_factor;
    int offset_w2 = offset2 % downscale_factor;

    int out_w2 = in_w * downscale_factor + offset_w2;
    int out_h2 = in_h * downscale_factor + offset_h2;
    int out_nh2 = in_n * out_H + out_h2;

    int2 out_pos2;
    out_pos2.x = out_w2 + (out_c2 / 4) * in_W;
    out_pos2.y = out_nh2;

    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos2);
    if (out_c2 % 4 == 0) {
        res.z = in.x;
    } else if (out_c2 % 4 == 1) {
        res.z = in.y;
    } else if (out_c2 % 4 == 2) {
        res.z = in.z;
    } else if (out_c2 % 4 == 3) {
        res.z = in.w;
    }

    int in_c3 = in_c4 * 4 + 3;
    int out_c3 = in_c3 / (downscale_factor * downscale_factor);
    int offset3 = in_c3 % (downscale_factor * downscale_factor);
    int offset_h3 = offset3 / downscale_factor;
    int offset_w3 = offset3 % downscale_factor;

    int out_w3 = in_w * downscale_factor + offset_w3;
    int out_h3 = in_h * downscale_factor + offset_h3;
    int out_nh3 = in_n * out_H + out_h3;

    int2 out_pos3;
    out_pos3.x = out_w3 + (out_c3 / 4) * in_W;
    out_pos3.y = out_nh3;

    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos3);
    if (out_c3 % 4 == 0) {
        res.w = in.x;
    } else if (out_c3 % 4 == 1) {
        res.w = in.y;
    } else if (out_c3 % 4 == 2) {
        res.w = in.z;
    } else if (out_c3 % 4 == 3) {
        res.w = in.w;
    }

    int2 in_pos;
    in_pos.x = in_c4 * (in_W / downscale_factor) + in_w;
    in_pos.y = in_nh;
    if (in_pos.x < out_W * ((out_C + 3) / 4) && in_pos.y < out_H * out_N) {
        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, in_pos, res);
    } 
}