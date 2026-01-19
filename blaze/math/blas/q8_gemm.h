//=================================================================================================
/*!
//  \file blaze/math/blas/q8_gemm.h
//  \brief Header file for q8_0 GEMM functionality
//
//  This file provides a minimal q8_0 (block-quantized int8) GEMM kernel intended for
//  f32 activations and q8_0 weights, with optional NEON acceleration.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_BLAS_Q8_GEMM_H_
#define _BLAZE_MATH_BLAS_Q8_GEMM_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace blaze {
namespace q8_0 {

struct block_q8_0 {
   std::uint16_t d;   // fp16 scale
   std::int8_t qs[32];
};

inline float fp16_to_fp32( std::uint16_t h ) noexcept
{
   __fp16 tmp;
   std::memcpy( &tmp, &h, sizeof( tmp ) );
   return static_cast<float>( tmp );
}

#if defined(__ARM_NEON)
inline float reduce_f32x4( float32x4_t v ) noexcept
{
#if defined(__aarch64__)
   return vaddvq_f32( v );
#else
   float32x2_t sum = vadd_f32( vget_low_f32( v ), vget_high_f32( v ) );
   sum = vpadd_f32( sum, sum );
   return vget_lane_f32( sum, 0 );
#endif
}
#endif

inline float dot_f32_q8_0( const float* a, const block_q8_0* b, std::size_t blocks ) noexcept
{
   float sum = 0.0f;

#if defined(__ARM_NEON)
   for( std::size_t blk = 0; blk < blocks; ++blk ) {
      const block_q8_0& qb = b[blk];
      const float scale = fp16_to_fp32( qb.d );
      float32x4_t acc = vdupq_n_f32( 0.0f );
      const std::int8_t* qs = qb.qs;
      const float* ap = a + blk * 32;

      for( int i = 0; i < 32; i += 16 ) {
         int8x16_t q = vld1q_s8( qs + i );

         int16x8_t ql = vmovl_s8( vget_low_s8( q ) );
         int16x8_t qh = vmovl_s8( vget_high_s8( q ) );

         int32x4_t ql0 = vmovl_s16( vget_low_s16( ql ) );
         int32x4_t ql1 = vmovl_s16( vget_high_s16( ql ) );
         int32x4_t qh0 = vmovl_s16( vget_low_s16( qh ) );
         int32x4_t qh1 = vmovl_s16( vget_high_s16( qh ) );

         float32x4_t f0 = vmulq_n_f32( vcvtq_f32_s32( ql0 ), scale );
         float32x4_t f1 = vmulq_n_f32( vcvtq_f32_s32( ql1 ), scale );
         float32x4_t f2 = vmulq_n_f32( vcvtq_f32_s32( qh0 ), scale );
         float32x4_t f3 = vmulq_n_f32( vcvtq_f32_s32( qh1 ), scale );

         float32x4_t a0 = vld1q_f32( ap + i + 0 );
         float32x4_t a1 = vld1q_f32( ap + i + 4 );
         float32x4_t a2 = vld1q_f32( ap + i + 8 );
         float32x4_t a3 = vld1q_f32( ap + i + 12 );

         acc = vmlaq_f32( acc, a0, f0 );
         acc = vmlaq_f32( acc, a1, f1 );
         acc = vmlaq_f32( acc, a2, f2 );
         acc = vmlaq_f32( acc, a3, f3 );
      }

      sum += reduce_f32x4( acc );
   }
#else
   for( std::size_t blk = 0; blk < blocks; ++blk ) {
      const block_q8_0& qb = b[blk];
      const float scale = fp16_to_fp32( qb.d );
      const float* ap = a + blk * 32;
      for( int i = 0; i < 32; ++i ) {
         sum += ap[i] * ( scale * static_cast<float>( qb.qs[i] ) );
      }
   }
#endif

   return sum;
}

// Row-major A (m x k) times row-quantized B (n x k), producing C (m x n).
// B is stored as q8_0 blocks per row: ldb is number of q8_0 blocks per row (k/32).
inline void sgemm_nt( std::size_t m, std::size_t n, std::size_t k,
                      float alpha, const float* A, std::size_t lda,
                      const block_q8_0* B, std::size_t ldb,
                      float beta, float* C, std::size_t ldc )
{
   assert( A != nullptr && B != nullptr && C != nullptr );
   assert( ( k % 32 ) == 0 );
   const std::size_t blocks = k / 32;

   for( std::size_t i = 0; i < m; ++i ) {
      const float* arow = A + i * lda;
      float* crow = C + i * ldc;
      for( std::size_t j = 0; j < n; ++j ) {
         const block_q8_0* brow = B + j * ldb;
         const float dot = dot_f32_q8_0( arow, brow, blocks );
         crow[j] = alpha * dot + beta * crow[j];
      }
   }
}

} // namespace q8_0
} // namespace blaze

#endif
