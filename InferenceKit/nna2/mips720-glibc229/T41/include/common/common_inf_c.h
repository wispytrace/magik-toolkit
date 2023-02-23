/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : common_inf_c.h
 * Authors     : lzwang
 * Create Time : 2022-05-23 10:45:25 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_INF_C_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_INF_C_H__
#include "common_def.h"

VENUS_C_API int magik_venus_init(int size);
VENUS_C_API int magik_venus_deinit();
VENUS_C_API int magik_venus_check();

#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_INF_C_H__ */
