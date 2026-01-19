//=================================================================================================
/*!
//  \file blaze/math/Q8Matrix.h
//  \brief Header file for q8_0 custom matrix aliases
*/
//=================================================================================================

#ifndef _BLAZE_MATH_Q8MATRIX_H_
#define _BLAZE_MATH_Q8MATRIX_H_

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/CustomMatrix.h>
#include <blaze/math/blas/q8_gemm.h>

namespace blaze {

template< AlignmentFlag AF = unaligned
        , PaddingFlag PF   = unpadded
        , bool SO          = rowMajor
        , typename Tag     = Group0 >
using Q8_0Matrix = CustomMatrix< q8_0::block_q8_0, AF, PF, SO, Tag >;

} // namespace blaze

#endif
