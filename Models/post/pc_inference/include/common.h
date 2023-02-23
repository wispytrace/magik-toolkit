/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : application/app.cc
 * Authors     : ffzhou/lqwang
 * Create Time : Mon 16 Nov 2020 04:56:00 PM CST
 *
 */

#ifndef INCLUDE_COMMON_H_
#define INCLUDE_COMMON_H_

#include <map>
#include <string>
#include <vector>

namespace magik {
namespace transformkit {
class GraphDef;
namespace magikexecutor {

class Tensor;

typedef struct {
    std::string input_name;
    std::string input_path;
    std::vector<int> input_shapes;
    float *img;
} input_info_t;

using StringTensorMapType = std::map<std::string, Tensor *>;

GraphDef *load_graph(const std::string &modelPath);

StringTensorMapType load_inputs(const std::vector<input_info_t> input_infos);

}  // namespace magikexecutor
}  // namespace transformkit
}  // namespace magik

#endif /* INCLUDE_COMMON_H_ */
