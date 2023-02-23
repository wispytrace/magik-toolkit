/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : venus.h
 * Authors     : klyu
 * Create Time : 2020-10-27 10:48:19 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_VENUS_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_VENUS_H__
#include "common/common_inf.h"
#include "core/tensor.h"
#include "core/type.h"
#include "utils/all.h"
#include <memory>
#include <vector>

namespace magik {
namespace venus {
class VENUS_API BaseNet {
public:
    BaseNet();
    virtual ~BaseNet() = 0;
    virtual int load_model(const char *model_path, bool memory_model = false, int start_off = 0);
    virtual int get_forward_memory_size(size_t &memory_size);
    /*memory must be alloced by nmem_memalign, and should be aligned with 64 bytes*/
    virtual int set_forward_memory(void *memory);
    /*free all memory except for input tensors, when smem_mode=DEFAULT or SMART_REUSE_MEM*/
    virtual int free_forward_memory();
    /*free memory of input tensors, when smem_mode=DEFAULT or SMART_REUSE_MEM*/
    virtual int free_inputs_memory();
    /*set internal memory management status, when smem_mode=ALL_SEPARABLE_MEM or SMART_REUSE_MEM
     *status=true: memory of input Tensors maybe free by BaseNet, be careful, data pointer should be
     *checked before read/write status=false: memory of input Tensors is managed by user
     */
    virtual void set_internal_mm_status(bool status);
    /*get internal memory management status*/
    virtual bool get_internal_mm_status();
    virtual void set_profiler_per_frame(bool status = false);
    virtual std::unique_ptr<Tensor> get_input(int index);
    virtual std::unique_ptr<Tensor> get_input_by_name(std::string &name);
    virtual std::vector<std::string> get_input_names();
    virtual std::unique_ptr<const Tensor> get_output(int index);
    virtual std::unique_ptr<const Tensor> get_output_by_name(std::string &name);
    virtual std::vector<std::string> get_output_names();
    /*get output names of given step*/
    virtual std::vector<std::string> get_output_names_step(int step);
    /*get color channel layout of input weight from model if set*/
    virtual ChannelLayout get_input_channel_layout(std::string &name);
    /*set color channel layout of input image for run network from NV12 data_fmt
     *pelease set same channel layout with model
     */
    virtual void set_input_channel_layout(std::string name, ChannelLayout layout);

    /*do inference, get all outputs*/
    virtual int run();
    /*get number of steps*/
    virtual int steps();
    /*do inference, get outputs for each step*/
    virtual int run_step();
};

/*
 * create inference handle.
 * input_data_fmt: NHWC or NV12
 */
VENUS_API std::unique_ptr<BaseNet> net_create(TensorFormat input_data_fmt = TensorFormat::NHWC,
                                              ShareMemoryMode smem_mode = ShareMemoryMode::DEFAULT);
VENUS_API int venus_lock();
VENUS_API int venus_unlock();
VENUS_API uint32_t venus_get_version_info();
VENUS_API uint32_t venus_get_used_mem_size();
} // namespace venus
} // namespace magik

namespace venus = magik::venus;
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_VENUS_H__ */
