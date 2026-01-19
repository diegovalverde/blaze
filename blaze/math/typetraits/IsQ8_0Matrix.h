//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsQ8_0Matrix.h
//  \brief Header file for the IsQ8_0Matrix type trait
*/
//=================================================================================================

#ifndef _BLAZE_MATH_TYPETRAITS_ISQ8_0MATRIX_H_
#define _BLAZE_MATH_TYPETRAITS_ISQ8_0MATRIX_H_

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/CustomMatrix.h>
#include <blaze/math/expressions/DMatTransExpr.h>
#include <blaze/math/blas/q8_gemm.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/RemoveCVRef.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

template< typename T >
struct IsQ8_0MatrixImpl
   : public FalseType
{};

template< AlignmentFlag AF, PaddingFlag PF, bool SO, typename Tag >
struct IsQ8_0MatrixImpl< CustomMatrix< q8_0::block_q8_0, AF, PF, SO, Tag > >
   : public TrueType
{};

template< AlignmentFlag AF, PaddingFlag PF, bool SO, typename Tag, typename RT >
struct IsQ8_0MatrixImpl< CustomMatrix< q8_0::block_q8_0, AF, PF, SO, Tag, RT > >
   : public TrueType
{};

template< typename T >
struct IsQ8_0Matrix
   : public IsQ8_0MatrixImpl< RemoveCVRef_t<T> >
{};

template< typename T >
constexpr bool IsQ8_0Matrix_v = IsQ8_0Matrix<T>::value;

template< typename T >
struct IsQ8_0TransExprImpl
   : public FalseType
{};

template< typename MT, bool SO >
struct IsQ8_0TransExprImpl< DMatTransExpr<MT,SO> >
   : public IsQ8_0Matrix<MT>
{};

template< typename T >
struct IsQ8_0TransExpr
   : public IsQ8_0TransExprImpl< RemoveCVRef_t<T> >
{};

template< typename T >
constexpr bool IsQ8_0TransExpr_v = IsQ8_0TransExpr<T>::value;

} // namespace blaze

#endif
