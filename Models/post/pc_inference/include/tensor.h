/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : tensor.h
 * Authors    : lqwang
 * Create Time: 2021-08-31:11:49:55
 * Description:
 *
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__
#include <iostream>

#include <map>
#include <string>
#include <vector>

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define MAGIK_API __attribute__((dllexport))
#else
#define MAGIK_API __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define MAGIK_API __attribute__((dllimport))
#else
#define MAGIK_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define MAGIK_API __attribute__((visibility("default")))
#else
#define MAGIK_API
#endif
#endif

namespace magik {
namespace transformkit {
namespace magikexecutor {

class MAGIK_API Tensor final {
public:
    typedef enum {
        DT_UNDEFINED = 0,
        DT_FLOAT = 1,
        DT_INT32 = 3,
        DT_UINT8 = 4,
        DT_INT16 = 5,
        DT_INT8 = 6,
        DT_UINT16 = 17,
        DT_UINT32 = 22
    } DataType;
    typedef enum { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU } SyncedHead;
    std::map<DataType, int> data_type_byte_size{{DT_UNDEFINED, 0}, {DT_FLOAT, 4}, {DT_UINT8, 1}, {DT_INT8, 1},
                                                {DT_INT16, 2},     {DT_INT32, 4}, {DT_UINT32, 4}};
    Tensor();
    Tensor(const Tensor &);
    Tensor(const std::vector<int> &shape, DataType data_type, void *buffer = nullptr, std::string head_at = "cpu");
    ~Tensor();

    SyncedHead head() const;

    void set_data(const std::vector<int> &shape, DataType data_type, void *buffer = nullptr);
    void set_gpu_data(const std::vector<int> &shape, DataType data_type, void *buffer = nullptr);
    inline int total() const {
        return total_;
    }
    inline void set_name(const std::string &name) {
        name_ = name;
    }
    inline const std::string &name(void) const {
        return name_;
    }

    const std::vector<Tensor *> &get_slave_tensors() const;
    std::vector<Tensor *> &get_mu_slave_tensors();

    void set_slave_tensors(Tensor *);

    template <typename T>
    inline const T &data(int index) const {
        if (total() <= index) {
            std::cout << "Index out of range error:" << name() << " " << total() << " " << index << std::endl;
            abort();
        }
        _to_cpu();
        T *buffer = static_cast<T *>(cpu_buffer_);
        const T &value = buffer[index];
        return value;
    }

    template <typename T>
    inline T &mutable_data(int index) {
        if (total() <= index) {
            std::cout << "Index out of range error:" << name() << " " << total() << " " << index << std::endl;
            abort();
        }
        _to_cpu();
        T *buffer = static_cast<T *>(cpu_buffer_);
        T &value = buffer[index];
        return value;
    }

    inline const std::vector<int> &shape() const {
        return shape_;
    }
    inline int dims() const {
        return shape_.size();
    }
    inline int byte_size() const {
        return byte_size_;
    }
    const void *get_ptr() const;
    const void *get_gpu_ptr() const;
    void *get_mutable_ptr();
    void *get_mutable_gpu_ptr();
    inline DataType data_type() const {
        return data_type_;
    }
    void reshape(const std::vector<int> &shape);
    Tensor &operator=(const Tensor &rhs);
    void to_gpu();
    void to_cpu();

    Tensor &fill(float alpha);

    Tensor add(const Tensor &other) const;
    Tensor mul(const Tensor &other) const;
    Tensor sub(const Tensor &other) const;
    Tensor div(const Tensor &other) const;

    Tensor sub(float other) const;
    Tensor mul(float other) const;
    Tensor add(float other) const;
    Tensor div(float other) const;

    Tensor less_than(float other) const;
    Tensor greater_than(float other) const;
    Tensor equal(float other) const;
    Tensor less_equal(float other) const;
    Tensor greater_equal(float other) const;
    Tensor bool_or(float other) const;
    Tensor bool_and(float other) const;

    Tensor less_than(const Tensor &other) const;
    Tensor greater_than(const Tensor &other) const;
    Tensor equal(const Tensor &other) const;
    Tensor less_equal(const Tensor &other) const;
    Tensor greater_equal(const Tensor &other) const;
    Tensor bool_or(const Tensor &other) const;
    Tensor bool_and(const Tensor &other) const;
    Tensor bool_not() const;

    Tensor sign() const;

    Tensor neg() const;
    Tensor abs() const;
    Tensor sigmoid() const;
    Tensor floor() const;
    Tensor log() const;
    Tensor pow(float alpha) const;
    Tensor exp(float alpha) const;
    Tensor clip(float min, float max) const;
    Tensor reduce_min(int axis, bool keep_dim = false) const;
    Tensor reduce_max(int axis, bool keep_dim = false) const;
    Tensor reduce_sum(int axis, bool keep_dim = false) const;
    Tensor reduce_mean(int axis, bool keep_dim = false) const;
    Tensor reduce_std(int axis, bool keep_dim = false) const;

    float reduce_min() const;
    float reduce_max() const;
    float reduce_sum() const;
    float reduce_mean() const;
    float reduce_std() const;

    Tensor &operator+=(const Tensor &other);
    Tensor &operator+=(float other);
    Tensor &operator-=(const Tensor &other);
    Tensor &operator-=(float other);
    Tensor &operator*=(const Tensor &other);
    Tensor &operator*=(float other);
    Tensor &operator/=(const Tensor &other);
    Tensor &operator/=(float other);
    inline Tensor operator-() const {
        return neg();
    }
    inline Tensor operator!() const {
        return bool_not();
    }

private:
    void _to_gpu() const;
    void _to_cpu() const;
    void _release();
    void _release_gpu();
    void _release_cpu();
    int _get_ele_byte_size(DataType data_type) const;

private:
    std::vector<Tensor *> slave_tensors_;
    std::string name_;
    std::vector<int> shape_;
    void *cpu_buffer_;
    void *gpu_buffer_;
    int byte_size_;
    int total_;
    bool own_cpu_data_;
    bool own_gpu_data_;
    SyncedHead head_;
    DataType data_type_;
    bool cpu_malloc_use_cuda_;
};

Tensor operator+(const Tensor &x, const Tensor &y);
Tensor operator+(const Tensor &x, float y);
Tensor operator+(float x, const Tensor &y);

Tensor operator-(const Tensor &x, const Tensor &y);
Tensor operator-(const Tensor &x, float y);
Tensor operator-(float x, const Tensor &y);

Tensor operator*(const Tensor &x, const Tensor &y);
Tensor operator*(const Tensor &x, float y);
Tensor operator*(float x, const Tensor &y);

Tensor operator/(const Tensor &x, const Tensor &y);
Tensor operator/(const Tensor &x, float y);
Tensor operator/(float x, const Tensor &y);

Tensor operator<(const Tensor &x, const Tensor &y);
Tensor operator<(const Tensor &x, float y);
Tensor operator<(float x, const Tensor &y);

Tensor operator<=(const Tensor &x, const Tensor &y);
Tensor operator<=(const Tensor &x, float y);
Tensor operator<=(float x, const Tensor &y);

Tensor operator>(const Tensor &x, const Tensor &y);
Tensor operator>(const Tensor &x, float y);
Tensor operator>(float x, const Tensor &y);

Tensor operator>=(const Tensor &x, const Tensor &y);
Tensor operator>=(const Tensor &x, float y);
Tensor operator>=(float x, const Tensor &y);

Tensor operator==(const Tensor &x, const Tensor &y);
Tensor operator==(const Tensor &x, float y);
Tensor operator==(float x, const Tensor &y);

Tensor operator&&(const Tensor &x, const Tensor &y);
Tensor operator&&(const Tensor &x, float y);
Tensor operator&&(float x, const Tensor &y);

Tensor operator||(const Tensor &x, const Tensor &y);
Tensor operator||(const Tensor &x, float y);
Tensor operator||(float x, const Tensor &y);

}  // namespace magikexecutor
}  // namespace transformkit
}  // namespace magik

#endif /* __TENSOR_H__ */
