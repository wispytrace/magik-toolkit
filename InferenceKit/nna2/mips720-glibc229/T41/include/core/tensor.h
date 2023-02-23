/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : tensor.h
 * Authors     : klyu
 * Create Time : 2020-08-04 14:58:26 (CST)
 * Description : tensor of api
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_CORE_TENSOR_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_CORE_TENSOR_H__

#include "type.h"
#include <initializer_list>
#include <stdint.h>
#include <vector>

namespace magik {
namespace venus {
using shape_t = std::vector<int32_t>;
class VENUS_API Tensor {
public:
    Tensor(shape_t shape, TensorFormat fmt = TensorFormat::NHWC);
    Tensor(std::initializer_list<int32_t> shape, TensorFormat fmt = TensorFormat::NHWC);
    /*data must be alloced by nmem_memalign, and should be aligned with 64 bytes*/
    Tensor(void *data, size_t bytes_size, TensorFormat fmt = TensorFormat::NHWC);
    Tensor(const Tensor &t);
    Tensor(void *tsx);       /*for internal*/
    Tensor(const void *tsx); /*for internal*/
    virtual ~Tensor();

    shape_t shape() const;
    DataType data_type() const;
    void reshape(shape_t &shape) const;
    void reshape(std::initializer_list<int32_t> shape) const;
    template <typename T>
    const T *data() const;
    template <typename T>
    T *mudata() const;
    void free_data() const;
    int set_data(void *data, size_t bytes_size);
    void *get_tsx() const; /*for internal*/
    int step(int dim) const;

private:
    void *tensorx;
    int *ref_count;
};
} // namespace venus
} // namespace magik

#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_CORE_TENSOR_H__ */
