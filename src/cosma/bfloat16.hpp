/**
 * @file bfloat16.hpp
 * @brief BFloat16 (Brain Floating Point) type definition
 * @author David Sanftenberg
 * @date 2025-10-19
 *
 * Implements the BFloat16 format: 16-bit floating point with 1 sign bit,
 * 8 exponent bits, and 7 mantissa bits. This format is compatible with
 * FP32's exponent range but has reduced precision, making it suitable for
 * deep learning and scientific computing where memory bandwidth is critical.
 *
 * Memory layout (big-endian bit ordering):
 * [15]: Sign bit
 * [14:7]: Exponent (8 bits, same as FP32)
 * [6:0]: Mantissa (7 bits, truncated from FP32's 23 bits)
 *
 * COSMA uses COSTA's bfloat16 implementation to avoid circular dependencies
 * and ensure consistency across both libraries.
 */

#pragma once

#include <costa/bfloat16.hpp>

namespace cosma {

// Use COSTA's bfloat16 implementation to avoid circular dependencies
using bfloat16 = costa::bfloat16;

// Re-export the abs function for convenience
using costa::abs;

} // namespace cosma
