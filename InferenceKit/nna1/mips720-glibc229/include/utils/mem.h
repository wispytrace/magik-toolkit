/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : mem.h
 * Authors     : klyu
 * Create Time : 2021-06-11 09:20:33 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_MEM_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_MEM_H__
#include "core/type.h"

namespace magik {
namespace venus {

/*alloc some memory from nmem*/
VENUS_API void *nmem_malloc(unsigned int size);

/*alloc some memory from nmem, and aligning the pointer with align*/
VENUS_API void *nmem_memalign(unsigned int align, unsigned int size);

/*free some memory alloced by nmem_malloc and nmem_memalign*/
VENUS_API void nmem_free(void *ptr);

VENUS_API void memcopy(void *dst, void *src, int n);

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_MEM_H__ */
