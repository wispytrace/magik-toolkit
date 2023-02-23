/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : postproc.h
 * Authors     : klyu
 * Create Time : 2020-12-24 15:02:22 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_POSTPROC_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_POSTPROC_H__
#include "core/type.h"
#include "venus.h"

namespace magik {
namespace venus {
/*
 * post-process NMS
 * type=0: hard nms, type=1: soft nms
 */
VENUS_API void nms(std::vector<ObjBbox_t> &input, std::vector<ObjBbox_t> &output,
                   float nms_threshold = 0.3, NmsType type = NmsType::HARD_NMS);
/*
 * post-process Generate Candidate Boxes
 * features : input feature
 * candidate_boxes : output candidate boxes
 * img_w,img_h : image width,height
 * classes : class number
 * box_num : the boxes number of each point
 * box_score_threshold : filter boxes based on box_score
 * multiply_class_score : -1: default, if calsses>1, score of classes is multiplied by results, else
 * not 0:  score of classes is not multiplied by results 1:  score of classes is multiplied by
 * results
 */
VENUS_API void generate_box(std::vector<Tensor> &features, std::vector<float> &strides,
                            std::vector<float> &anchor, std::vector<ObjBbox_t> &candidate_boxes,
                            int img_w, int img_h, int classes, int box_num,
                            float box_score_threshold,
                            DetectorType detector_type = DetectorType::YOLOV3,
                            int multiply_class_score = -1);
/*
 *
 * input0 : input feature0
 * input1 : input feature1
 * output1 : output feature
 */
VENUS_API void mat_mul(Tensor &input0, Tensor &input1, Tensor &output);
/*
 * input : output tensor
 * out_boxes : output boxes
 */
VENUS_API void get_objbox_from_output(Tensor &input, std::vector<ObjBbox_t> &out_boxes);
} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_POSTPROC_H__ */
