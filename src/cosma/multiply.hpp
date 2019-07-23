#pragma once

#include <cosma/communicator.hpp>
#include <cosma/context.hpp>
#include <cosma/interval.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <cosma/multiply_using_layout.hpp>

namespace cosma {

template <typename Scalar>
void multiply(context &ctx,
              CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              const Strategy &strategy,
              MPI_Comm comm,
              Scalar alpha,
              Scalar beta);

template <typename Scalar>
void multiply(context &ctx,
              CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              Interval &m,
              Interval &n,
              Interval &k,
              Interval &P,
              size_t step,
              const Strategy &strategy,
              communicator &comm,
              Scalar alpha,
              Scalar beta);

template <typename Scalar>
void sequential(context &ctx,
                CosmaMatrix<Scalar> &A,
                CosmaMatrix<Scalar> &B,
                CosmaMatrix<Scalar> &C,
                Interval &m,
                Interval &n,
                Interval &k,
                Interval &P,
                size_t step,
                const Strategy &strategy,
                communicator &comm,
                Scalar alpha,
                Scalar beta);

template <typename Scalar>
void parallel(context &ctx,
              CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              Interval &m,
              Interval &n,
              Interval &k,
              Interval &P,
              size_t step,
              const Strategy &strategy,
              communicator &comm,
              Scalar alpha,
              Scalar beta);

} // namespace cosma
