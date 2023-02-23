/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : args_parser.h
 * Authors    : lqwang
 * Create Time: 2021-04-08:11:24:29
 * Description:
 *
 */

#ifndef __ARGSPARSER_H__
#define __ARGSPARSER_H__

#include <set>
#include <string>
#include <vector>
#include "cmdline.h"
namespace magik {
namespace transformkit {
namespace magikexecutor {

class ArgsParser {
public:
    ArgsParser(int argc, char **argv);

    std::string getModelPath();
    std::vector<std::string> getInputNames();
    std::string getOpName();
    std::vector<std::string> getInputPaths();
    std::vector<std::vector<int>> getInputSizes();

private:
    std::vector<std::string> _split(const std::string &str, std::set<char> splitChars = {','});

private:
    int argc_;
    char **argv_;
    cmdline::parser cmdParser;
};

}  // namespace magikexecutor
}  // namespace transformkit
}  // namespace magik
#endif /* __ARGSPARSER_H__ */
