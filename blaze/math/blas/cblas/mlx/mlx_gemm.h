#ifndef MLX_GEMM_H_INCLUDED
#define MLX_GEMM_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <mlx/mlx.h>   // MLX C++ header :contentReference[oaicite:0]{index=0}

namespace mx = mlx::core;

// sgemm_mlx.cpp
#include <algorithm>
#include <cassert>
#include <cstddef>

#include "mlx/mlx.h"

namespace mx = mlx::core;

// Row-major SGEMM with leading dimensions:
//   A: [m x k], row-major, leading dim lda (lda >= k)
//   B: [k x n], row-major, leading dim ldb (ldb >= n)
//   C: [m x n], row-major, leading dim ldc (ldc >= n)
// Computes: C = alpha * (A @ B) + beta * C
inline void mlx_sgemm(
    int m, int n, int k,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc)
{
    using namespace mx;
    set_default_device(Device::gpu);

    assert(m > 0 && n > 0 && k > 0);
    assert(A != nullptr && B != nullptr && C != nullptr);
    assert(lda >= k);
    assert(ldb >= n);
    assert(ldc >= n);

    // Optional: enforce GPU
    // set_default_device(Device::gpu);


    // Allocate contiguous MLX arrays for the math shapes.
    namespace mx = mlx::core;

    // Shape is typically an alias for std::vector<int64_t>
    mx::array a = mx::zeros(mx::Shape{m, k}, mx::float32);
    mx::array b = mx::zeros(mx::Shape{k, n}, mx::float32);
    mx::array c = mx::zeros(mx::Shape{m, n}, mx::float32);



    assert(a.data<float>() != nullptr);
    assert(b.data<float>() != nullptr);
    assert(c.data<float>() != nullptr);



// make sure it’s contiguous & evaluated
a = mx::contiguous(a);
b = mx::contiguous(b);
c = mx::contiguous(c);
assert( a.flags().row_contiguous); //Make sure or else we get a segfault
assert( b.flags().row_contiguous);
assert( c.flags().row_contiguous);
mx::eval(a); //We need this or else it does not "materialize" and can't call  a.data<float>();
mx::eval(b);
mx::eval(c);


    // Pack A (m x k) from row-major with lda into contiguous 'a'
    
        float* a_data = a.data<float>();
        for (std::size_t i{0}; i < m; ++i) {
            const float* src_row = A + i * lda;
            float* dst_row       = a_data + i * k;
            std::copy(src_row, src_row + k, dst_row);
        }

    // Pack B (k x n) from row-major with ldb into contiguous 'b'{
        float* b_data = b.data<float>();
        for (std::size_t i{0}; i < k; ++i) {
            const float* src_row = B + i * ldb;
            float* dst_row       = b_data + i * n;
            std::copy(src_row, src_row + n, dst_row);
        }
    


    // Pack C (m x n) from row-major with ldc into contiguous 'c'
    
        float* c_data = c.data<float>();
        for (std::size_t i{0}; i < m; ++i) {
            const float* src_row = C + i * ldc;
            float* dst_row       = c_data + i * n;
            std::copy(src_row, src_row + n, dst_row);
        }
    


    // Matrix multiply in MLX: [m x k] @ [k x n] -> [m x n]
    mx::array prod = mx::matmul(a, b);

    // Combine alpha and beta
    mx::array c_out = prod;
    if (beta == 0.0f) {
        if (alpha == 1.0f) {
            c_out = prod;
        } else {
            c_out = alpha * prod;
        }
    } else {
        if (alpha == 1.0f) {
            c_out = prod + beta * c;
        } else {
            c_out = alpha * prod + beta * c;
        }
    }


    // Ensure computation finishes before reading back

     auto t0 = std::chrono::high_resolution_clock::now();

    mx::eval(c_out);

     auto t1 = std::chrono::high_resolution_clock::now();

            // std::cerr << "METAL CORE Gemm took "
            // << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
            // << " milliseconds!\n";




    // Unpack back into C using ldc
    {
        const float* src = c_out.data<float>();
        for (int i = 0; i < m; ++i) {
            const float* src_row = src + std::size_t(i) * n;
            float* dst_row       = C + std::size_t(i) * ldc;
            std::copy(src_row, src_row + n, dst_row);
        }
    }


}



#endif

