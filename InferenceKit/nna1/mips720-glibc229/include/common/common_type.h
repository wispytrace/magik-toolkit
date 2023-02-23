/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : common_type.h
 * Authors     : lzwang
 * Create Time : 2021-10-14 18:14:03 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_TYPE_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_TYPE_H__
#include "common_def.h"

namespace magik {
namespace venus {
enum class ShareMemoryMode : int VENUS_API {
    DEFAULT = 0, /*SHARE_ONE_NETWORK*/
    SHARE_ONE_THREAD = 1,
    SET_FROM_EXTERNAL = 2, /*NOT-SUPPORTED*/
    ALL_SEPARABLE_MEM = 3, /*NO_SHARE, DYNAMIC MANAGE, FOR INTERNAL*/
    SMART_REUSE_MEM = 4,   /*SMART REUSE MEMORY, FOR INTERNAL*/
};

enum class ChannelLayout : int VENUS_API {
    NONE = -1,
    NV12 = 0,
    BGRA = 1,
    RGBA = 2,
    ARGB = 3,
    RGB = 4,
    GRAY = 5,
    FP = 6
};

enum class TransformType : int VENUS_API {
    NONE = -1,
    NV12_NV12 = 0,
};

enum class DataType : int VENUS_API {
    NONE = -1, /*is not supported*/
    AUTO = 0,
    FP32 = 1,
    UINT32 = 2,
    INT32 = 3,
    UINT16 = 4,
    INT16 = 5,
    UINT8 = 6,
    INT8 = 7,
    UINT6B = 8,  /*is not supported*/
    UINT6BP = 9, /*is not supported*/
    UINT4B = 10,
    UINT2B = 11,
    UINT10 = 12,
    INT10 = 13,
    UINT12 = 14,
    INT12 = 15,
    UINT14 = 16, /*is not supported*/
    INT14 = 17,  /*is not supported*/
    BOOL = 18,
};

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_TYPE_H__ */
