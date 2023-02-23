/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : common_def.h
 * Authors     : lzwang
 * Create Time : 2022-05-23 11:30:24 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_DEF_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_DEF_H__

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define DLL_EXPORTS __attribute__((dllexport))
#else
#define DLL_EXPORTS __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define DLL_EXPORTS __attribute__((dllimport))
#else
#define DLL_EXPORTS __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define DLL_EXPORTS __attribute__((visibility("default")))
#else
#define DLL_EXPORTS
#endif
#endif

#ifndef VENUS_EXTERN_C
#ifdef __cplusplus
#define VENUS_EXTERN_C extern "C"
#else
#define VENUS_EXTERN_C
#endif
#endif
#include <stdint.h>

#define VENUS_API DLL_EXPORTS
#define VENUS_C_API VENUS_EXTERN_C DLL_EXPORTS

#if defined __VENUS_EXPORT_DLL__
#define VENUS_EXPORT DLL_EXPORTS
#else
#define VENUS_EXPORT
#endif

#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_DEF_H__ */
