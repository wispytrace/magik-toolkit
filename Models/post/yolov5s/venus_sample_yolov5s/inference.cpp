/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : inference.cc
 * Authors     : ffzhou
 * Create Time : 2022-07-16 09:22:44 (CST)
 * Description :
 *
 */
#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
static const uint8_t color[3] = {0xff, 0, 0};

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
#include <algorithm>

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef VENUS_PROFILE
#define RUN_CNT 10
#else
#define RUN_CNT 1
#endif

#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

using namespace std;
using namespace magik::venus;

struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};

uint8_t* read_bin(const char* path)
{
    std::ifstream infile;
    infile.open(path, std::ios::binary | std::ios::in);
    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    uint8_t* buffer_pointer = new uint8_t[length];
    infile.read((char*)buffer_pointer, length);
    infile.close();
    return buffer_pointer;
}

std::vector<std::string> splitString(std::string srcStr, std::string delimStr,bool repeatedCharIgnored = false)
{
    std::vector<std::string> resultStringVector;
    std::replace_if(srcStr.begin(), srcStr.end(), [&](const char& c){if(delimStr.find(c)!=std::string::npos){return true;}else{return false;}}, delimStr.at(0));
    size_t pos=srcStr.find(delimStr.at(0));
    std::string addedString="";
    while (pos!=std::string::npos) {
        addedString=srcStr.substr(0,pos);
        if (!addedString.empty()||!repeatedCharIgnored) {
            resultStringVector.push_back(addedString);
        }
        srcStr.erase(srcStr.begin(), srcStr.begin()+pos+1);
        pos=srcStr.find(delimStr.at(0));
    }
    addedString=srcStr;
    if (!addedString.empty()||!repeatedCharIgnored) {
        resultStringVector.push_back(addedString);
    }
    return resultStringVector;
}

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
    for(int i = 0; i < (int)in_boxes.size(); i++) {
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

#ifdef VENUS_DEBUG
    int ret = 0;
    if (argc != 3)
    {
        printf("%s model_path image_bin\n", argv[0]);
        exit(0);
    }
    std::string model_path = argv[1];
    std::string image_bin = argv[2];

    uint8_t* imagedata = read_bin(image_bin.c_str());
	std::vector<std::string> result_str = splitString(splitString(image_bin, ".")[0],"_");
    int vec_size = result_str.size();

    int n = atoi(result_str[vec_size - 4].c_str());
    int in_h = atoi(result_str[vec_size - 3].c_str());
    int in_w = atoi(result_str[vec_size - 2].c_str());
    int c = atoi(result_str[vec_size - 1].c_str());
    printf("image_bin shape:%d %d %d %d\n", n, in_h, in_w, c);

    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }

    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(TensorFormat::NHWC);
    ret = test_net->load_model(model_path.c_str());

    input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);
    int data_cnt = 1;
    for (auto i : input->shape()) 
    {
        std::cout << i << ",";
        data_cnt *= i;
    }
    std::cout << std::endl;

	for (int i = 0; i < in_h; i ++)
	{
		for (int j = 0; j < in_w; j++)
		{
			indata[i*in_w*4 + j*4 + 0] = imagedata[i*in_w*3 + j*3 + 0];
			indata[i*in_w*4 + j*4 + 1] = imagedata[i*in_w*3 + j*3 + 1];
			indata[i*in_w*4 + j*4 + 2] = imagedata[i*in_w*3 + j*3 + 2];
			indata[i*in_w*4 + j*4 + 3] = 0;
		}
	}

    test_net->run();

#else

    int ret = 0;
    if (argc != 3)
    {
        printf("%s model_path img_path\n", argv[0]);
        exit(0);
    }

    int ori_img_h = -1;
    int ori_img_w = -1;
    float scale = 1.0;
    int in_w = 640, in_h = 384;

    PixelOffset pixel_offset;
    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(TensorFormat::NHWC);

    std::string model_path = argv[1];
    ret = test_net->load_model(model_path.c_str());

    std::string image_path = argv[2];
    int comp = 0;
    unsigned char *imagedata = stbi_load(argv[2], &ori_img_w, &ori_img_h, &comp, 3); // image format is bgra

    magik::venus::shape_t temp_inshape;
    temp_inshape.push_back(1);
    temp_inshape.push_back(ori_img_h);
    temp_inshape.push_back(ori_img_w);
    temp_inshape.push_back(4);
    venus::Tensor input_tensor(temp_inshape);
    uint8_t *temp_indata = input_tensor.mudata<uint8_t>();

	for (int i = 0; i < ori_img_h; i ++)
	{
		for (int j = 0; j < ori_img_w; j++)
		{
			temp_indata[i*ori_img_w*4 + j*4 + 0] = imagedata[i*ori_img_w*3 + j*3 + 2];
			temp_indata[i*ori_img_w*4 + j*4 + 1] = imagedata[i*ori_img_w*3 + j*3 + 1];
			temp_indata[i*ori_img_w*4 + j*4 + 2] = imagedata[i*ori_img_w*3 + j*3 + 0];
			temp_indata[i*ori_img_w*4 + j*4 + 3] = 0;
		}
	}
    Img img = {
        .w = ori_img_w,
        .h = ori_img_h,
        .c = 3,
        .w_stride = ori_img_w*3,
        .data = imagedata
    };

//    magik::venus::memcopy((void*)temp_indata, (void*)(imagedata), src_size * sizeof(uint8_t));

    printf("ori_image w,h: %d ,%d \n",ori_img_w,ori_img_h);
    input = test_net->get_input(0);
    magik::venus::shape_t input_shape = input->shape();
    printf("model-->%d ,%d %d \n",input_shape[1], input_shape[2], input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
//    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);

    float scale_x = (float)in_w/(float)ori_img_w;
    float scale_y = (float)in_h/(float)ori_img_h;
    scale = scale_x < scale_y ? scale_x:scale_y;  //min scale
    printf("scale---> %f\n",scale);
    int valid_dst_w = (int)(scale*ori_img_w);
    if (valid_dst_w % 2 == 1)
        valid_dst_w = valid_dst_w + 1;
    int valid_dst_h = (int)(scale*ori_img_h);
    if (valid_dst_h % 2 == 1)
    {
        valid_dst_h = valid_dst_h + 1;
    }

    int dw = in_w - valid_dst_w;
    int dh = in_h - valid_dst_h;
    
    pixel_offset.top = int(round(float(dh)/2 - 0.1));
    pixel_offset.bottom = int(round(float(dh)/2 + 0.1));
    pixel_offset.left = int(round(float(dw)/2 - 0.1));
    pixel_offset.right = int(round(float(dw)/2 + 0.1));
    
    check_pixel_offset(pixel_offset);
    printf("resize padding over: \n");
    printf("resize valid_dst, w:%d h %d\n",valid_dst_w,valid_dst_h);
    printf("padding info top :%d bottom %d left:%d right:%d \n",pixel_offset.top,pixel_offset.bottom,pixel_offset.left,pixel_offset.right);


    magik::venus::BsExtendParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::SYMMETRY;
    param.in_layout = magik::venus::ChannelLayout::RGBA;
    param.out_layout = magik::venus::ChannelLayout::RGBA;
    warp_resize(input_tensor, *input, &param);

#ifdef TIME
    struct timeval tv; 
    uint64_t time_last;
    double time_ms;
#endif

#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec;
#endif
    for(int i = 0 ; i < RUN_CNT; i++)
    {
        test_net->run();
    }
#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    time_ms = time_last*1.0/1000;
    printf("test_net run time_ms:%fms\n", time_ms);
#endif


#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec;
#endif

    std::unique_ptr<const venus::Tensor> out0 = test_net->get_output(0);
    std::unique_ptr<const venus::Tensor> out1 = test_net->get_output(1);
    std::unique_ptr<const venus::Tensor> out2 = test_net->get_output(2);

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
#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    time_ms = time_last*1.0/1000;
	printf("post net time_ms:%fms\n", time_ms);
#endif

    for (int i = 0; i < int(output_boxes.size()); i++) 
    {
        auto person = output_boxes[i];
        printf("box:   ");
        printf("%d ",(int)person.box.x0);
        printf("%d ",(int)person.box.y0);
        printf("%d ",(int)person.box.x1);
        printf("%d ",(int)person.box.y1);
        printf("%.2f ",person.score);

        Point pt1 = {
            .x = (int)person.box.x0,
            .y = (int)person.box.y0
        };
        Point pt2 = {
            .x = (int)person.box.x1,
            .y = (int)person.box.y1
        };
        sample_draw_box_for_image(&img, pt1, pt2, color, 2);
        printf("\n");
    }
    stbi_write_bmp("result.bmp", ori_img_w, ori_img_h, 3, img.data);// w h
    free(img.data);


    ret = venus::venus_deinit();
    if (0 != ret) 
    {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
#endif

}

void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h)
{
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


