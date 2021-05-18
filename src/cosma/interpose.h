/*
 * taken from: https://github.com/helixhorned/interpose/tree/feature/c-header-and-example-logger
 * which is forked from: https://github.com/ccurtsinger/interpose
 *
 * MIT License

   Copyright (c) 2017 Charlie Curtsinger

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#if !defined(__INTERPOSE_H)
#define __INTERPOSE_H

#include <stdint.h>

#include <dlfcn.h>

#if defined(__ELF__)

#if !defined _GNU_SOURCE
# error Must define _GNU_SOURCE for RTLD_NEXT to be available.
#endif

// The C version of the interposition macro differs from that of the C++ version: the user
// has to provide the argument type-and-name list as a macro argument (instead of postfixing
// it to the macro invocation), and to redundantly provide the list of just argument names,
// which must match those in the type-and-name list.
#define INTERPOSE__C_GENERIC__(RETURN_TYPE, NAME, ARG_TYPE_AND_NAME_LIST, ...) \
  static RETURN_TYPE Real__##NAME ARG_TYPE_AND_NAME_LIST { \
    static __typeof__(NAME)* real_##NAME; \
    __typeof__(NAME)* func = __atomic_load_n(&real_##NAME, __ATOMIC_CONSUME); \
    if(!func) { \
      func = (__typeof__(NAME)*)( \
        (uintptr_t)(dlsym(RTLD_NEXT, #NAME))); \
      __atomic_store_n(&real_##NAME, func, __ATOMIC_RELEASE); \
    } \
    __VA_ARGS__; \
  } \
  extern __typeof__(NAME) NAME __attribute__((weak, alias("__interpose_" #NAME))); \
  extern RETURN_TYPE __interpose_##NAME ARG_TYPE_AND_NAME_LIST

#define INTERPOSE_C(RETURN_TYPE, NAME, ARG_TYPE_AND_NAME_LIST, ARG_NAME_LIST) \
  INTERPOSE__C_GENERIC__(RETURN_TYPE, NAME, ARG_TYPE_AND_NAME_LIST, return func ARG_NAME_LIST)

#define INTERPOSE_C_VOID(NAME, ARG_TYPE_AND_NAME_LIST, ARG_NAME_LIST) \
  INTERPOSE__C_GENERIC__(void, NAME, ARG_TYPE_AND_NAME_LIST, func ARG_NAME_LIST)

#elif defined(__APPLE__)

/// Structure exposed to the linker for interposition
struct __osx_interpose {
	const void* new_func;
	const void* orig_func;
};

/**
 * Generate a macOS interpose struct
 * Types from: http://opensource.apple.com/source/dyld/dyld-210.2.3/include/mach-o/dyld-interposing.h
 */
#define OSX_INTERPOSE_STRUCT(NEW, OLD) \
  static const struct __osx_interpose __osx_interpose_##OLD \
    __attribute__((used, section("__DATA, __interpose"))) = \
    { (const void*)((uintptr_t)(&(NEW))), \
      (const void*)((uintptr_t)(&(OLD))) }

/**
  * The OSX interposition process is much simpler. Just create an OSX interpose struct,
  * include the actual function in the `real` namespace, and declare the beginning of the
  * replacement function with the appropriate return type.
  */
#define INTERPOSE__C_GENERIC__(RETURN_TYPE, NAME, ARG_TYPE_AND_NAME_LIST, ...) \
  static RETURN_TYPE Real__##NAME ARG_TYPE_AND_NAME_LIST { \
    __VA_ARGS__; \
  } \
  extern RETURN_TYPE __interpose_##NAME ARG_TYPE_AND_NAME_LIST; \
  OSX_INTERPOSE_STRUCT(__interpose_##NAME, NAME); \
  extern RETURN_TYPE __interpose_##NAME ARG_TYPE_AND_NAME_LIST

#define INTERPOSE_C(RETURN_TYPE, NAME, ARG_TYPE_AND_NAME_LIST, ARG_NAME_LIST) \
  INTERPOSE__C_GENERIC__(RETURN_TYPE, NAME, ARG_TYPE_AND_NAME_LIST, return NAME ARG_NAME_LIST)

#define INTERPOSE_C_VOID(NAME, ARG_TYPE_AND_NAME_LIST, ARG_NAME_LIST) \
  INTERPOSE__C_GENERIC__(void, NAME, ARG_TYPE_AND_NAME_LIST, NAME ARG_NAME_LIST)

#else
# error Unsupported platform.
#endif

#endif
