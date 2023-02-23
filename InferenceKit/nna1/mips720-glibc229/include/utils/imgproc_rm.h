#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_RM_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_RM_H__

#include "common/common_utils.h"
#include "core/tensor.h"
#include "core/type.h"
#include "imgproc_type.h"
#include <vector>
namespace magik {
namespace venus {

/*
 * resize tensor
 * input: input tensor, format bgra
 * output: output tensor, format bgra
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int warp_resize_bgra(const Tensor &input, Tensor &output, PaddingType padtype,
                               uint8_t padval);
/*
 * resize tensor
 * input: input tensor, format nv12
 * output: output tensor
 * cvtbgra: if true, ouput format is bgra, else nv12
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int warp_resize_nv12(const Tensor &input, Tensor &output, bool cvtbgra,
                               PaddingType padtype, uint8_t padval);
/*
 * crop and resize tensor
 * input: input tensor, format bgra
 * output: output tensor
 * boxes: boxes for input tensor crop
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int crop_resize_bgra(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Bbox_t> &boxes, PaddingType padtype, uint8_t padval);
/*
 * crop and resize tensor
 * input: input tensor, format nv12
 * output: output tensor
 * boxes: boxes for input tensor crop
 * cvtbgra: if true, ouput format is bgra, else nv12
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int crop_resize_nv12(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Bbox_t> &boxes, bool cvtbgra, PaddingType padtype,
                               uint8_t padval);
/*
 * resize input nv12 format data
 * input: address of input nv12 format data
 * output: output tensor
 * img_h: input height
 * img_w: input width
 * line_stride: input line stride(byte)
 * cvtbgra: if true, ouput format is bgra, else nv12
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 */
VENUS_API int common_resize_nv12(const void *input, Tensor &output, int img_h, int img_w,
                                 int line_stride, bool cvtbgra, PaddingType padtype, uint8_t padval,
                                 AddressLocate input_locate);
/*
 * affine tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: affine matrix tensor
 */
VENUS_API int warp_affine_bgra(const Tensor &input, Tensor &output, Tensor &matrix);

/*
 * affine tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: tensor of affine matrix
 * cvtbgra: if true, ouput format is bgra, else nv12
 */
VENUS_API int warp_affine_nv12(const Tensor &input, Tensor &output, Tensor &matrix, bool cvtbgra);

/*
 * crop and affine tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: affine matrix tensor
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_affine_bgra(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes);

/*
 * crop and affine tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: tensor of affine matrix
 * cvtbgra: if true, ouput format is bgra, else nv12
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_affine_nv12(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Tensor> &matrix, bool cvtbgra,
                               std::vector<Bbox_t> &boxes);

/*
 * perspective tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: perspective matrix tensor
 */
VENUS_API int warp_perspective_bgra(const Tensor &input, Tensor &output, Tensor &matrix);

/*
 * perspective tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: perspective matrix tensor
 * cvtbgra: if true, ouput format is bgra, else nv12
 */
VENUS_API int warp_perspective_nv12(const Tensor &input, Tensor &output, Tensor &matrix,
                                    bool cvtbgra);
/*
 * crop and perspective tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: perspective matrix tensor
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_perspective_bgra(const Tensor &input, std::vector<Tensor> &output,
                                    std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes);

/*
 * crop and perspective tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: perspective matrix tensor
 * cvtbgra: if true, ouput format is bgra, else nv12
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_perspective_nv12(const Tensor &input, std::vector<Tensor> &output,
                                    std::vector<Tensor> &matrix, bool cvtbgra,
                                    std::vector<Bbox_t> &boxes);
} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_RM_H__ */
