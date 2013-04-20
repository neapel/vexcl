#ifndef VEXCL_SAT_HPP
#define VEXCL_SAT_HPP

/*
The MIT License

Copyright (c) 2012-2013 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   sat.hpp
 * \author Pascal Germroth <pascal@ensieve.org>
 * \brief  Summed-Area Tables, ported from Andre Maximo et al.'s gpufilter.
 *         see http://code.google.com/p/gpufilter/
 */

/* gpufilter is under the MIT License (see above)
 * Copyright (c) 2011, Diego Nehab, Andre Maximo, Rodolfo S. Lima and Hugues Hoppe
 */

#include <vexcl/vector.hpp>

namespace vex {


template <class F>
struct sat_expr
    : vector_expression< boost::proto::terminal< additive_vector_transform >::type >
{
    F &f;
    const vector<typename F::value_type> &input;

    sat_expr(F &f, const vector<typename F::value_type> &x) : f(f), input(x) {}

    template <bool negate, bool append>
    void apply(vector<typename F::value_type> &output) const {
        static_assert(!negate, "Doesn't support negation.");
        static_assert(!append, "Doesn't support appending.");
        f(input(0), output(0));
    }
};


/// Summed-area table.
/**
 * From a vector with elements data[x + y * width] this computes
 * a summed-area table sat[x + y * width] = sum(data[x' + y' * width], 0 <= x' <= x, 0 <= y' <= y).
 *
 * Usage:
 * \code
 * vector<double> data(ctx, width * height), out(ctx, width * height);
 * SAT<double> sat(ctx, width, height);
 * out = sat(data);  // out-of-place
 * data = sat(data); // in-place
 * \endcode
 */
template<class T>
struct SAT {
    typedef T value_type;

    const size_t half_warp_size = 16; //(HWS) Half Warp Size
    const size_t max_threads = 192; //(MTS) Maximum number of threads per block with 8 blocks per SM
    const size_t max_warps = 6; //(MW) Maximum number of warps per block with 8 blocks per SM (with all warps computing)
    const size_t opt_warps = 5; //(SOW) Dual-scheduler optimized number of warps per block (with 8 blocks per SM and to use the dual scheduler with 1 computing warp)
    const size_t warp_size = 32; //(WS) Warp size (defines b x b block size where b = warp_size)

    const std::vector<cl::CommandQueue> &queues;
    cl::Kernel stage1, stage2, stage3, stage4;
    size_t in_width, in_height, width, height, m_size, n_size;
    bool reformat;
    cl::Buffer inout, ybar, vhat, ysum;

    SAT(const std::vector<cl::CommandQueue> &queues, size_t w, size_t h)
    : queues(queues), in_width(w), in_height(h) {
        if(queues.size() != 1)
            throw std::invalid_argument("Only single device context supported.");
        width = in_width;
        height = in_height;
        if(width % warp_size != 0) width += warp_size - (width % warp_size);
        if(height % warp_size != 0) height += warp_size - (height % warp_size);
        reformat = in_width != width || in_height != height;
        m_size = (width + warp_size - 1) / warp_size;
        n_size = (height + warp_size - 1) / warp_size;
        auto context = qctx(queues[0]);
        if(reformat) inout = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * width * height);
        ybar = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * n_size * width);
        vhat = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * m_size * height);
        ysum = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * m_size * n_size);
        make_kernels();
    }

    sat_expr<SAT<T>> operator()(const vector<T> &in) {
        return sat_expr<SAT<T>>(*this, in);
    }

    void operator()(const cl::Buffer in, cl::Buffer out) {
        cl::size_t<3> off, region;
        off[0] = off[1] = off[2] = 0;
        region[0] = sizeof(T) * in_width; region[1] = in_height; region[2] = 1;
        if(reformat) queues[0].enqueueCopyBufferRect(in, inout, off, off, region, 0, 0, sizeof(T) * width, 0);

        auto img_range = cl::NDRange(m_size * warp_size, n_size * opt_warps);
        auto cybar = cl::NDRange(((width + max_threads - 1) / max_threads) * warp_size, 1 * max_warps);
        auto cvhat = cl::NDRange(1 * warp_size, ((height + max_threads - 1) / max_threads) * max_warps);
        auto ws_sow = cl::NDRange(warp_size, opt_warps);
        auto ws_mw = cl::NDRange(warp_size, max_warps);

        stage1.setArg(0, reformat ? inout : in);
        stage1.setArg(1, ybar);
        stage1.setArg(2, vhat);
        stage1.setArg<cl_uint>(3, width);
        stage1.setArg<cl_uint>(4, height);
        queues[0].enqueueNDRangeKernel(stage1, cl::NullRange, img_range, ws_sow);

        stage2.setArg(0, ybar);
        stage2.setArg(1, ysum);
        stage2.setArg<cl_uint>(2, width);
        stage2.setArg<cl_uint>(3, n_size);
        stage2.setArg<cl_uint>(4, m_size);
        queues[0].enqueueNDRangeKernel(stage2, cl::NullRange, cybar, ws_mw);

        stage3.setArg(0, ysum);
        stage3.setArg(1, vhat);
        stage3.setArg<cl_uint>(2, height);
        stage3.setArg<cl_uint>(3, m_size);
        queues[0].enqueueNDRangeKernel(stage3, cl::NullRange, cvhat, ws_mw);

        stage4.setArg(0, reformat ? inout : out);
        stage4.setArg(1, reformat ? inout : in);
        stage4.setArg(2, ybar);
        stage4.setArg(3, vhat);
        stage4.setArg<cl_uint>(4, width);
        stage4.setArg<cl_uint>(5, height);
        queues[0].enqueueNDRangeKernel(stage4, cl::NullRange, img_range, ws_sow);

        if(reformat) queues[0].enqueueCopyBufferRect(inout, out, off, off, region, sizeof(T) * width, 0, 0, 0);
    }

    void make_kernels() {
        std::ostringstream o;
        o << standard_kernel_header(qdev(queues[0]))
          << "typedef " << type_name<T>() << " T;\n"
          << "#define warp_size " << warp_size << "\n"
          << "#define opt_warps " << opt_warps << "\n"
          << "#define max_warps " << max_warps << "\n"
          << "#define half_warp_size " << half_warp_size << "\n"
          << // Stage 1: reads input and computes incomplete prologues ybar (sum of rows) and vhat (sum of columns)
             "__attribute__((reqd_work_group_size(warp_size, opt_warps, 1)))\n"
             "kernel void stage1(global const T *in, global T *ybar, global T *vhat, uint width, uint height) {\n"
             "    const size_t tx = get_local_id(0), ty = get_local_id(1),\n"
             "        bx = get_group_id(0), by = get_group_id(1),\n"
             "        col = bx * warp_size + tx, row0 = by * warp_size;\n"
             "    local T block[warp_size][warp_size + 1];\n"
             "    {\n"
             "        size_t lx = tx, ly = ty;\n"
             "        in += (row0 + ty) * width + col;\n"
             "        ybar += by * width + col;\n"
             "        vhat += bx * height + row0 + tx;\n"
             "        for(size_t i = 0 ; i < warp_size - (warp_size % opt_warps) ; i += opt_warps) {\n"
             "            block[ly][lx] = *in;\n"
             "            ly += opt_warps;\n"
             "            in += opt_warps * width;\n"
             "        }\n"
             "        if(ty < warp_size % opt_warps) block[ly][lx] = *in;\n"
             "    }\n"
             "    barrier(CLK_LOCAL_MEM_FENCE);\n"
             "    if(ty == 0) {\n"
             "        {\n"
             "            size_t ly = 0;\n"
             "            T prev = block[ly++][tx];\n"
             "            for(size_t i = 1 ; i < warp_size ; i++)\n"
             "                prev = block[ly++][tx] += prev;\n"
             "            *ybar = prev;\n"
             "        }{\n"
             "            size_t lx = 0;\n"
             "            T prev = block[tx][lx++];\n"
             "            for(size_t i = 1 ; i < warp_size ; i++)\n"
             "                prev += block[tx][lx++];\n"
             "            *vhat = prev;\n"
             "        }\n"
             "    }\n"
             "}\n"
             // Stage 2: complete prologues ybar (cumsum), compute scalars ysum (sum of ybar)
             "__attribute__((reqd_work_group_size(warp_size, max_warps, 1)))\n"
             "kernel void stage2(global T *ybar, global T *ysum, uint width, uint n_size, uint m_size) {\n"
             "    const size_t tx = get_local_id(0), ty = get_local_id(1),\n"
             "        bx = get_group_id(0), col0 = bx*max_warps+ty, col = col0*warp_size+tx;\n"
             "    if(col >= width) return;\n"
             "    ybar += col;\n"
             "    T y = *ybar;\n"
             "    const size_t ln = half_warp_size + tx;\n"
             "    if(tx == warp_size - 1) ysum += col0;\n"
             "    local volatile T block[max_warps][half_warp_size + warp_size + 1];\n"
             "    if(tx < half_warp_size) block[ty][tx] = (T)0;\n"
             "    else block[ty][ln] = (T)0;\n"
             "    for(size_t n = 1 ; n < n_size ; ++n) { // ysum\n"
             "        block[ty][ln] = y;\n"
             "        block[ty][ln] += block[ty][ln-1];\n"
             "        block[ty][ln] += block[ty][ln-2];\n"
             "        block[ty][ln] += block[ty][ln-4];\n"
             "        block[ty][ln] += block[ty][ln-8];\n"
             "        block[ty][ln] += block[ty][ln-16];\n"
             "        if(tx == warp_size-1) {\n"
             "            *ysum = block[ty][ln];\n"
             "            ysum += m_size;\n"
             "        }\n"
             "        // fix ybar -> y\n"
             "        ybar += width;\n"
             "        y = *ybar += y;\n"
             "    }\n"
             "}\n"
             // Stage 3: complete prologues vhat (cumsum), add scalar to each.
             "__attribute__((reqd_work_group_size(warp_size, max_warps, 1)))\n"
             "kernel void stage3(global const T *ysum, global T *vhat, uint height, uint m_size) {\n"
             "    const size_t tx = get_local_id(0), ty = get_local_id(1),\n"
             "        by = get_group_id(1), row0 = by*max_warps+ty, row = row0*warp_size+tx;\n"
             "    if(row >= height) return;\n"
             "    vhat += row;\n"
             "    T y = (T)0, v = (T)0;\n"
             "    if(row0 > 0) ysum += (row0 - 1) * m_size;\n"
             "    for(size_t m = 0 ; m < m_size ; m++) { // fix vhat -> v\n"
             "        if(row0 > 0) y = *ysum++;\n"
             "        v = *vhat += v + y;\n"
             "        vhat += height;\n"
             "    }\n"
             "}\n"
             // Stage 4: add y to each row, v to each column, cumsum => done.
             "__attribute__((reqd_work_group_size(warp_size, opt_warps, 1)))\n"
             "kernel void stage4(global T *out, global const T *in, global const T *y, global const T *v, uint width, uint height) {\n"
             "    const bool inplace = out == in;\n"
             "    const size_t tx = get_local_id(0), ty = get_local_id(1),\n"
             "        bx = get_group_id(0), by = get_group_id(1),\n"
             "        col = bx*warp_size+tx, row0 = by*warp_size;\n"
             "    local T block[warp_size][warp_size + 1];\n"
             "    {\n"
             "        size_t lx = tx, ly = ty;\n"
             "        in += (row0 + ty) * width + col;\n"
             "        if(by > 0) y += (by - 1) * width + col;\n"
             "        if(bx > 0) v += (bx - 1) * height + row0 + tx;\n"
             "        for(size_t i = 0 ; i < warp_size - (warp_size % opt_warps) ; i += opt_warps) {\n"
             "            block[ly][lx] = *in;\n"
             "            ly += opt_warps;\n"
             "            in += opt_warps * width;\n"
             "        }\n"
             "        if(ty < warp_size % opt_warps) block[ly][lx] = *in;\n"
             "    }\n"
             "    barrier(CLK_LOCAL_MEM_FENCE);\n"
             "    if(ty == 0) {\n"
             "        {\n"
             "            size_t lx = tx, ly = 0;\n"
             "            T prev;\n"
             "            if(by > 0) prev = *y;\n"
             "            else prev = (T)0;\n"
             "            for(size_t i = 0 ; i < warp_size ; i++, ly++)\n"
             "                block[ly][lx] = prev = block[ly][lx] + prev;\n"
             "        }\n"
             "        {\n"
             "            size_t lx = 0, ly = tx;\n"
             "            T prev;\n"
             "            if(bx > 0) prev = *v;\n"
             "            else prev = (T)0;\n"
             "            for(size_t i = 0 ; i < warp_size ; i++, lx++)\n"
             "                block[ly][lx] = prev = block[ly][lx] + prev;\n"
             "        }\n"
             "    }\n"
             "    barrier(CLK_LOCAL_MEM_FENCE);\n"
             "    {\n"
             "        size_t lx = tx, ly = ty;\n"
             "        if(inplace) out = in - (warp_size - (warp_size % opt_warps)) * width;\n"
             "        else out += (row0 + ty) * width + col;\n"
             "        for(size_t i = 0 ; i < warp_size - (warp_size % opt_warps) ; i += opt_warps) {\n"
             "            *out = block[ly][lx];\n"
             "            ly += opt_warps;\n"
             "            out += opt_warps * width;\n"
             "        }\n"
             "        if(ty < warp_size % opt_warps) *out = block[ly][lx];\n"
             "    }\n"
             "}\n";
        auto program = build_sources(qctx(queues[0]), o.str());
        stage1 = cl::Kernel(program, "stage1");
        stage2 = cl::Kernel(program, "stage2");
        stage3 = cl::Kernel(program, "stage3");
        stage4 = cl::Kernel(program, "stage4");
    }
};


}

#endif
