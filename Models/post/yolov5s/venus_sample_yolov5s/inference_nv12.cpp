/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : model_run.cc
 * Authors     : klyu
 * Create Time : 2020-10-28 12:22:44 (CST)
 * Description :
 *
 */

#include "venus.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)


void write_output_bin(const float* out_ptr, int size)
{
    std::string out_name = "out_res.bin";
    std::ofstream owput;
    owput.open(out_name, std::ios::binary);
    if (!owput || !owput.is_open() || !owput.good()) {
        owput.close();
        return ;
    }
    owput.write((char *)out_ptr, size * sizeof(float));
    owput.close();
	return ;
}

using namespace std;
using namespace magik::venus;

typedef struct
{
    unsigned char* image;  
    int w;
    int h;
}input_info_t;

struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};


void check_pixel_offset(PixelOffset &pixel_offset){
    // 5 5 -> 6 4
    // padding size not is Odd number
    if(pixel_offset.top % 2 == 1){
        pixel_offset.top += 1;
        pixel_offset.bottom -=1;
    }
    if(pixel_offset.left % 2 == 1){
        pixel_offset.left += 1;
        pixel_offset.right -=1;
    }
}

void trans_coords(std::vector<magik::venus::ObjBbox_t> &in_boxes, PixelOffset &pixel_offset,float scale){
    
    printf("pad_x:%d pad_y:%d scale:%f \n",pixel_offset.left,pixel_offset.top,scale);
    for(int i = 0; i < in_boxes.size(); i++) {

        in_boxes[i].box.x0 = (in_boxes[i].box.x0 - pixel_offset.left) / scale;
        in_boxes[i].box.x1 = (in_boxes[i].box.x1 - pixel_offset.left) / scale;
        in_boxes[i].box.y0 = (in_boxes[i].box.y0 - pixel_offset.top) / scale;
        in_boxes[i].box.y1 = (in_boxes[i].box.y1 - pixel_offset.top) / scale;
    }
}


void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h);
void manyclass_nms(std::vector<magik::venus::ObjBbox_t> &input, std::vector<magik::venus::ObjBbox_t> &output, int classnums, int type, float nms_threshold);

int main(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
            printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc != 5)
    {
        printf("%s model_path nv12_path w h\n", argv[0]);
        exit(0);
    }


    bool cvtbgra;
    cvtbgra = true;
    void *handle = NULL;

    int ori_img_h;
    int ori_img_w;
    float scale;
    int in_w = 640, in_h = 384;

    
    PixelOffset pixel_offset;
    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    std::unique_ptr<venus::BaseNet> test_net;
    if (cvtbgra){
        test_net = venus::net_create(TensorFormat::NHWC);
    }
    else { 
        test_net = venus::net_create(TensorFormat::NV12);
    }

    std::string model_path = argv[1];
    ret = test_net->load_model(model_path.c_str());

//    char*ptr = (char*) malloc(7623720);
//
//    FILE *fp1 = NULL;
//    fp1 = fopen(model_path.c_str(),"r");
//
//    char buffer[1024] = {0};
//    char* ptr1 = ptr;
//    int sum = 0;
//    while(!feof(fp1))
//    {
//        int count = fread(buffer, sizeof (char), sizeof(buffer), fp1);
//        memcpy(ptr1, buffer, sizeof(buffer));
//        ptr1 = ptr1 + sizeof(buffer);
//		sum = sum + count;
//    }
//    printf("sum:%d\n", sum);
//    ret = test_net->load_model(ptr, true);
//    free(ptr);


    input_info_t input_src;
    //
    FILE *fp = NULL;
    printf("-----------a--------------------\n");
    input_src.w = atoi(argv[3]);
    input_src.h = atoi(argv[4]);
    int nv12_size = input_src.w * input_src.h * 1.5;
    unsigned char* nv12_data = (unsigned char*)malloc(nv12_size);
    fp = fopen(argv[2], "rb");
    if(fp){
        int ret = fread(nv12_data, 1, nv12_size, fp);
        printf("\033[32mret %d\033[0m\n", ret);
        fclose(fp);
    }else{
        printf("open nv12 file failed\n");
        return -1;
    }
    input_src.image = (unsigned char*)nv12_data;
    
    //---------------------process-------------------------------
    // get ori image w h
    ori_img_w = input_src.w;
    ori_img_h = input_src.h;
    printf("ori_image w,h: %d ,%d \n",ori_img_w,ori_img_h);
    int line_stride = ori_img_w;
    input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    if (cvtbgra)
    {
        input->reshape({1, in_h, in_w , 4});
    }else
    {
        input->reshape({1, in_h, in_w, 1});
    }
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);
    //resize and padding
    magik::venus::Tensor temp_ori_input({1, ori_img_h, ori_img_w, 1}, TensorFormat::NV12);
    uint8_t *tensor_data = temp_ori_input.mudata<uint8_t>();
    int src_size = int(ori_img_h * ori_img_w * 1.5);
    magik::venus::memcopy((void*)tensor_data, (void*)input_src.image, src_size * sizeof(uint8_t));

    float scale_x = (float)in_w/(float)ori_img_w;
    float scale_y = (float)in_h/(float)ori_img_h;
    scale = scale_x < scale_y ? scale_x:scale_y;  //min scale
    printf("scale---> %f\n",scale);

    int valid_dst_w = (int)(scale*ori_img_w);
	if (valid_dst_w % 2 == 1)
		valid_dst_w = valid_dst_w + 1;
    int valid_dst_h = (int)(scale*ori_img_h);
	if (valid_dst_h % 2 == 1)
		valid_dst_h = valid_dst_h + 1;

    int dw = in_w - valid_dst_w;
    int dh = in_h - valid_dst_h;

    pixel_offset.top = int(round(float(dh)/2 - 0.1));
    pixel_offset.bottom = int(round(float(dh)/2 + 0.1));
    pixel_offset.left = int(round(float(dw)/2 - 0.1));
    pixel_offset.right = int(round(float(dw)/2 + 0.1));
    
//    check_pixel_offset(pixel_offset);

    magik::venus::BsCommonParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::SYMMETRY;
    param.input_height = ori_img_h;
    param.input_width = ori_img_w;
    param.input_line_stride = ori_img_w;
    param.in_layout = magik::venus::ChannelLayout::NV12;
    param.out_layout = magik::venus::ChannelLayout::RGBA;
    magik::venus::common_resize((const void*)tensor_data, *input.get(), magik::venus::AddressLocate::NMEM_VIRTUAL, &param);

    printf("resize padding over: \n");
    printf("resize valid_dst, w:%d h %d\n",valid_dst_w,valid_dst_h);
    printf("padding info top :%d bottom %d left:%d right:%d \n",pixel_offset.top,pixel_offset.bottom,pixel_offset.left,pixel_offset.right);

    test_net->run();

    std::unique_ptr<const venus::Tensor> out0 = test_net->get_output(0);
    std::unique_ptr<const venus::Tensor> out1 = test_net->get_output(1);
    std::unique_ptr<const venus::Tensor> out2 = test_net->get_output(2);

    std::unique_ptr<const venus::Tensor> out_tensor = test_net->get_output(0);
    const float *out_ptr = out_tensor->data<float>();
    write_output_bin(out_ptr, out_tensor->shape()[0] * out_tensor->shape()[1] * out_tensor->shape()[2] * out_tensor->shape()[3]);

    auto shape0 = out0->shape();
    auto shape1 = out1->shape();
    auto shape2 = out2->shape();

    int shape_size0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int shape_size1 = shape1[0] * shape1[1] * shape1[2] * shape1[3];
    int shape_size2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];

    venus::Tensor temp0(shape0);
    venus::Tensor temp1(shape1);
    venus::Tensor temp2(shape2);

    float* p0 = temp0.mudata<float>();
    float* p1 = temp1.mudata<float>();
    float* p2 = temp2.mudata<float>();

    memcopy((void*)p0, (void*)out0->data<float>(), shape_size0 * sizeof(float));
    memcopy((void*)p1, (void*)out1->data<float>(), shape_size1 * sizeof(float));
    memcopy((void*)p2, (void*)out2->data<float>(), shape_size2 * sizeof(float));
   
    std::vector<venus::Tensor> out_res;
    out_res.push_back(temp0);
    out_res.push_back(temp1);
    out_res.push_back(temp2);

    std::vector<magik::venus::ObjBbox_t>  output_boxes;
    output_boxes.clear();
    generateBBox(out_res, output_boxes, in_w, in_h);
    trans_coords(output_boxes, pixel_offset, scale);

    for (int i = 0; i < int(output_boxes.size()); i++) {
        auto person = output_boxes[i];
        printf("box:   ");
        printf("%d ",(int)person.box.x0);
        printf("%d ",(int)person.box.y0);
        printf("%d ",(int)person.box.x1);
        printf("%d ",(int)person.box.y1);
        printf("%.2f ",person.score);
        printf("\n");
    }

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}

void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h){

  float person_threshold = 0.3;
  int classes = 80;
  float nms_threshold = 0.6;
  std::vector<float> strides = {8.0, 16.0, 32.0};
  int box_num = 3;
  std::vector<float> anchor = {10,13,  16,30,  33,23, 30,61,  62,45,  59,119, 116,90,  156,198,  373,326};


  std::vector<magik::venus::ObjBbox_t>  temp_boxes;
  venus::generate_box(out_res, strides, anchor, temp_boxes, img_w, img_h, classes, box_num, person_threshold, magik::venus::DetectorType::YOLOV5);
//  venus::nms(temp_boxes, candidate_boxes, nms_threshold); 
  manyclass_nms(temp_boxes, candidate_boxes, classes, 0, nms_threshold);

}

void manyclass_nms(std::vector<magik::venus::ObjBbox_t> &input, std::vector<magik::venus::ObjBbox_t> &output, int classnums, int type, float nms_threshold) {
  int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  std::vector<magik::venus::ObjBbox_t> classbuf;
  for (int clsid = 0; clsid < classnums; clsid++) {
    classbuf.clear();
    for (int i = 0; i < box_num; i++) {
      if (merged[i])
        continue;
      if(clsid!=input[i].class_id)
        continue;
      classbuf.push_back(input[i]);
      merged[i] = 1;

    }
    magik::venus::nms(classbuf, output, nms_threshold, magik::venus::NmsType::HARD_NMS);
  }
}


