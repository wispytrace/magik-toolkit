/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : utils.h
 * Authors     : klyu(kanglong.yu@ingenic.com)
 * Create Time : 2020-06-12 10:28:15 (CST)
 * Description :
 *
 */

#ifndef __JZDL_UTILS_H__
#define __JZDL_UTILS_H__
#include "mat.h"
#include <cmath>
#include <stdint.h>
#include <vector>

namespace jzdl {

/**************************TypeDef*******************************/
enum {
    JZDL_INTER_NEAREST = 0,  // nearest neighbor interpolation
    JZDL_INTER_LINEAR = 1,   // bilinear interpolation
    JZDL_INTER_CUBIC = 2,    // bicubic interpolation
    JZDL_INTER_AREA = 3,     // area-based (or super) interpolation
    JZDL_INTER_LANCZOS4 = 4, // Lanczos interpolation over 8x8 neighborhood
    JZDL_INTER_MAX = 7
};

typedef struct {
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
} ObjBbox_t;

typedef struct {
    int x0;
    int y0;
    int x1;
    int y1;
} Bbox_t;

typedef struct {
    float x;
    float y;
} Point2f;

typedef enum {
    PAD_CONSTANT = 0,
    PAD_REPLICATE = 1,
} PadType_t;

typedef struct {
    int top;
    int bottom;
    int left;
    int right;
    PadType_t type;
    uint8_t value;
} PadInfo_t;

/*****************************Functions****************************/
/*
 * image(Gray/BGR/RGB) resize
 */
JZDL_API void resize(const Mat<uint8_t> &_src, Mat<uint8_t> &_dst,
                     int interpolation = JZDL_INTER_LINEAR);
/*
 * post-process NMS
 * type=0: hard nms, type=1: soft nms
 */
JZDL_API void nms(std::vector<ObjBbox_t> &input, std::vector<ObjBbox_t> &output,
                  float nms_threshold = 0.3, int type = 0);

/*
 * input = input - beta
 */
JZDL_API void image_sub(Mat<uint8_t> &input, uint8_t beta);

/*
 * get_affine_transform: 3 pairs of points
 */
JZDL_API Mat<double> get_affine_transform(const Point2f src[], const Point2f dst[]);

/*
 * References
 * "Least-squares estimation of transformation parameters between two
 *  point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
 */
JZDL_API jzdl::Mat<double> similar_transform(jzdl::Mat<double> src, jzdl::Mat<double> dst);

/*
 * wrap_affine
 */
JZDL_API void wrap_affine(const Mat<uint8_t> &src, Mat<uint8_t> &dst, Mat<double> &M);

/*
 * bgra2bgr
 */
JZDL_API void bgra2bgr(const Mat<uint8_t> &src, Mat<uint8_t> &dst);

/*
 * rgba2rgb
 */
JZDL_API void rgba2rgb(const Mat<uint8_t> &src, Mat<uint8_t> &dst);

/*
 * bgra2rgb
 */
JZDL_API void bgra2rgb(const Mat<uint8_t> &src, Mat<uint8_t> &dst);

/*
 * rgba2bgr
 */
JZDL_API void rgba2bgr(const Mat<uint8_t> &src, Mat<uint8_t> &dst);

/*
 * output_parser: parse jzdl inference output result.
 */
JZDL_API std::vector<Mat<float> > output_parser(Mat<float> &outputs);

/*
 * post-process Sigmoid
 */
JZDL_API void sigmoid(Mat<float> &feat);

/*
 * image crop
 */
JZDL_API int get_roi(const Mat<uint8_t> &img, Mat<uint8_t> &roi, Bbox_t &box);
/*
 * image padding
 */
JZDL_API int image_pad(const Mat<uint8_t> &src, Mat<uint8_t> &dst, PadInfo_t &pad_info);

/*
 * vector and matrix cross product
 * vector_a = [a0, a1, a2]
 * mat_b = [b0, b1, b2,
 *          b3, b4, b5,
 *          b6, b7, b8]
 * vector_c = [a0*b0+a1*b3+a2*b6, a0*b1+a1*b4+a2*b7, a0*b2+a1*B5+a2*b8]
 */
JZDL_API int mat_mul(const jzdl::Mat<float> &vector_a, const jzdl::Mat<float> &mat_b,
                     jzdl::Mat<float> &vector_c);
} /*namespace jzdl*/
#endif /* __JZDL_UTILS_H__ */
