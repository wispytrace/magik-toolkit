/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : imgproc.h
 * Authors     : klyu
 * Create Time : 2020-12-24 14:59:36 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_H__
#include "common/common_utils.h"
#include "core/tensor.h"
#include "core/type.h"
#include "imgproc_rm.h"
#include <vector>
namespace magik {
namespace venus {

/*
 * color space conversion
 * input: input tensor, format nv12
 * output: output tensor, format bgra
 */
VENUS_API int warp_covert_nv2bgr(const Tensor &input, Tensor &output);

/*
 * similar transform
 * input_src : input tensor, source perspective matrix
 * input_dst : input tensor, dst perspective matrix
 * output : output tensor, similar transform result matrix
 * retval : 0, success, <0 failed
 */
VENUS_API int similar_transform(Tensor &input_src, Tensor &input_dst, Tensor &output);

/*
 * affine transform
 * input_src : input tensor, source affine matrix
 * input_dst : input tensor, dst affine matrix
 * output : output tensor, affine transform result matrix
 * transform_type:0, nv12 to nv12
 * retval : 0, success, <0 failed
 */
VENUS_API int get_affine_transform(Tensor &input_src, Tensor &input_dst, Tensor &output,
                                   TransformType transform_type);

/*
 * affine rotation matrix
 * angle : rotation angle
 * scale : enlarge or narrow scale
 * (center_w,center_h) : Rotate image center point
 * M : affine rotation result matrix
 */
VENUS_API int get_affine_rotation_matrix(float angle,float scale,int center_w,int center_h,float *M);

/*
 * perspective transform
 * input_src : input tensor, source perspective matrix
 * input_dst : input tensor, dst perspesctive matrix
 * output : output tensor,  transform result matrix
 * transform_type:0, nv12 to nv12
 * retval : 0, success, <0 failed
 */
VENUS_API int get_perspective_transform(Tensor &input_src, Tensor &input_dst, Tensor &output,
                                        TransformType transform_type);

/*********new version**************/
/*if input chn equal 4, output format must equal input format*/
/*
 * resize tensor
 * input: input tensor
 * output: output tensor
 * param: resize param
 */

VENUS_API int warp_resize(const Tensor &input, Tensor &output, BsExtendParam *param);

/*
 * crop and resize tensor
 * input: input tensor
 * output: output tensor
 * boxes: boxes for input tensor crop
 * param: resize param
 */
VENUS_API int crop_resize(const Tensor &input, std::vector<Tensor> &output,
                          std::vector<Bbox_t> &boxes, BsExtendParam *param);

/*
 * crop and resize input nv12 format data
 * input: address of input nv12 format data
 * output: output tensor
 * boxes: boxes for input nv12 format data crop
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: resize param
 */
VENUS_API int crop_common_resize(const void *input, std::vector<Tensor> &output,
                                 std::vector<Bbox_t> &boxes, AddressLocate input_locate,
                                 BsCommonParam *param);

/*
 * affine tensor
 * input: input tensor
 * output: output tensor
 * matrix: affine matrix tensor
 * param: affine param
 */
VENUS_API int warp_affine(const Tensor &input, Tensor &output, Tensor &matrix,
                          BsExtendParam *param);
/*
 * affine tensor
 * input: address of input nv12 or bgra format data
 * output: output tensor
 * matrix: affine matrix tensor
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: affine param
 */
VENUS_API int common_affine(const void *input, Tensor &output, Tensor &matrix,
                            AddressLocate input_locate, BsCommonParam *param);
/*
 * crop and affine tensor
 * input: input tensor
 * output: output tensor
 * matrix: affine matrix tensor
 * boxes: boxes for input tensor crop
 * param: affine param
 */
VENUS_API int crop_affine(const Tensor &input, std::vector<Tensor> &output,
                          std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes,
                          BsExtendParam *param);
/*
 * affine tensor
 * input: address of input nv12 or bgra format data
 * output: output tensor
 * boxes: boxes for input tensor crop
 * matrix: affine matrix tensor
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: affine param
 */
VENUS_API int crop_common_affine(const void *input, std::vector<Tensor> &output,
                                 std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes,
                                 AddressLocate input_locate, BsCommonParam *param);
/*
 * perspective tensor
 * input: input tensor
 * output: output tensor
 * matrix: perspective matrix tensor
 * param: perspective param
 */
VENUS_API int warp_perspective(const Tensor &input, Tensor &output, Tensor &matrix,
                               BsExtendParam *param);

/*
 * perspective tensor
 * input: address of input nv12 or bgra format data
 * output: output tensor
 * matrix: perspective matrix tensor
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: perspective param
 */
VENUS_API int common_perspective(const void *input, Tensor &output, Tensor &matrix,
                                 AddressLocate input_locate, BsCommonParam *param);
/*
 * perspective tensor
 * input: input tensor
 * output: output tensor
 * matrix: perspective matrix tensor
 * param: perspective param
 */
VENUS_API int crop_perspective(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes,
                               BsExtendParam *param);
/*
 * perspective tensor
 * input: address of input nv12 or bgra format data
 * output: output tensor
 * matrix: perspective matrix tensor
 * boxes: boxes for input tensor crop
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: perspective param
 */
VENUS_API int crop_common_perspective(const void *input, std::vector<Tensor> &output,std::vector<Tensor> &matrix,
                                      std::vector<Bbox_t> &boxes, AddressLocate input_locate,BsCommonParam *param);
/*
 * resize input nv12 format data
 * input: address of input nv12 format data
 * output: output tensor
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: resize param
 */
VENUS_API int common_resize(const void *input, Tensor &output, AddressLocate input_locate,
                            BsCommonParam *param);

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_H__ */
