/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : graph_executor.h
 * Authors    : lqwang
 * Create Time: 2021-04-07:16:16:44
 * Description:
 *
 */

#ifndef __GRAPHEXECUTOR_H__
#define __GRAPHEXECUTOR_H__

#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef ONGPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

#include "tensor.h"

namespace magik {
namespace transformkit {
class GraphDef;
class NodeDef;
namespace magikexecutor {
class OpExecutor;
class Graph;
class Tensor;
class NodeNameMap;

class MAGIK_API GraphExecutor {
public:
    typedef enum { NONE = -1, CPU = 0, GPU = 1 } DEVICE;
    GraphExecutor(const GraphDef *graph, DEVICE device = DEVICE::CPU);
    GraphExecutor(const std::string &magik_path, DEVICE device = DEVICE::CPU);
    GraphExecutor(const GraphDef *graph, const std::map<std::string, Tensor *> &inputs, DEVICE device = DEVICE::CPU);
    ~GraphExecutor();
    bool work(const std::vector<std::string> &from = {}, const std::string &to = "");
    DEVICE device() const;
    void set_inplace(bool inplace);
    bool inplace() const;
    void set_device(DEVICE device);
    void set_input(Tensor *tensor);
    void update_graph();
    void set_input(const std::string &tensor_name, Tensor *tensor);
    void set_inputs(const std::map<std::string, Tensor *> &inputs);
    const Tensor *get_node_tensor(const std::string &name) const;
    std::vector<std::string> get_output_names() const;
    std::vector<std::string> get_input_names() const;
    const Tensor *get_output_tensor(const std::string &name) const;
    Tensor *get_mutable_output_tensor(const std::string &name) const;
    std::vector<float> get_node_float_data(const Tensor *tensor) const;
    Tensor *get_mutable_node_tensor(const std::string &name) const;
    const std::vector<Tensor *> &get_node_weights(const std::string &name) const;
    OpExecutor *get_op_executor(const std::string &name) const;

    bool query_gpu_device() const;
    bool check_gpu_device(int device_id) const;
    int find_gpu_device(int start_id) const;
    bool set_gpu_device(int device_id);
#ifdef ONGPU
    cudnnHandle_t get_cudnn(void) const;
    cudaStream_t get_stream(void) const;
#endif
private:
    GraphExecutor(void);
    void _pre_run(const std::vector<std::string> &from = {}, const std::string &to = "");
    void _ir_graph_initialize(const GraphDef *graph);
    std::string _get_target_device(const GraphDef *graph);
    void _parse_node_weight(Tensor *weight, const NodeDef *const_node);

private:
    DEVICE device_;
    Graph *ir_graph_;
    std::vector<std::string> from_;
    std::string to_;
    const GraphDef *graph_;
    bool own_graph_;
    NodeNameMap *all_node_map_;
    friend class Graph;
#ifdef ONGPU
    cudnnHandle_t cudnn = nullptr;
    cudaStream_t stream = nullptr;
#endif
};

}  // namespace magikexecutor
}  // namespace transformkit
}  // namespace magik

#endif /* __GRAPHEXECUTOR_H__ */
