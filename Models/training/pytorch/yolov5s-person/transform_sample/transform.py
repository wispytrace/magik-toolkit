#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : transform.py
## Authors    : thtang
## Create Time: 2022-06-22:10:48:43
## Description:
##
##


import argparse
import magik_transformer.magik_transform as mt

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Input \'GraphDef\' file to load.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Magik saver file.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="Magik quant cfg file.")
    opt = parser.parse_args()
    return opt

def main(opt):
    mt.magik_transform(opt.model_file, opt.output_file, opt.config_file)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
