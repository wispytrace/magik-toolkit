#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include "net.h"
#include "img_input.h"
#include "fstream"
#include <iomanip>
#include "utils.h"
using namespace std;
using namespace jzdl;

#define det_size 416

#define MEM_MODEL
#ifdef MEM_MODEL
#include "magik_model_persondet.mk.h"
#endif



void generateBBox(float * p, vector<ObjBbox_t> & candidate_boxes, int img_w, int img_h);
void copyMakeBorder(jzdl::Mat<uint8_t> in, jzdl::Mat<uint8_t>& out, int top ,int bottom, int left, int right, int value);
void letter_box(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, int value);
void trans_coords(const jzdl::Mat<uint8_t>& img, jzdl::Mat<uint8_t>& working_img, vector<ObjBbox_t> in_boxes, vector<ObjBbox_t>& out_boxes);




int main(int argc, char* argv[]) {

    printf("imagedata size:%d\n", sizeof(image));

    jzdl::BaseNet *persondet = jzdl::net_create();


#ifdef MEM_MODEL
    persondet->load_model((const char*)magik_model_persondet_mk, true);
#else
    std::string model_file_path = "magik_model_persondet.bin";
    persondet->load_model(model_file_path.c_str());
#endif

    int input_index = persondet->get_model_input_index();
    int output_index = persondet->get_model_output_index();
    printf("input_index=%d, output_index=%d\n", input_index, output_index);
//    std::vector<uint32_t> input_shape = persondet->get_input_shape();
//    printf("input_shape.w=%d, input_shape.h=%d, input_shape.c=%d\n", input_shape[0], input_shape[1], input_shape[2]);


    jzdl::Mat<uint8_t> src(640, 480, 3, (uint8_t*)image);
    jzdl::Mat<uint8_t> dst;

    letter_box(src, dst, 128);

    image_sub(dst, 128);

    jzdl::Mat<int8_t> img(dst.w, dst.h, dst.c, (int8_t*)dst.data);

    jzdl::Mat<float> out;

    printf("########Input Done#######\n");

    persondet->input(img);
    persondet->run(out);

    ofstream InputFile_ori("img_feature_jzdl_ori.h");
    {
        for (int i = 0; i < out.h * out.w * out.c; i++)
        {
            float temp = static_cast<float>(out.data[i]);
    
            InputFile_ori  << temp;
            if ((i+1) % 1 == 0)
            {
                InputFile_ori << endl;
            }
        }
    }
//
//
//
//    std::vector<Mat<float>> res = output_parser(out);
//
//    ofstream InputFile("img_feature_jzdl.h");
//    for (int j = 0 ; j < res.size();j ++)
//    {
//        jzdl::Mat<float> out1 = res[j];
//        printf("%d_%d_%d\n", out1.w, out1.h, out1.c);
//
//        printf("%p\n", out1.data);
//        for (int i = 0; i < out1.h * out1.w * out1.c; i++)
//        {
//            float temp = static_cast<float>(out1.data[i]);
//    
//            InputFile  << temp;
//            if ((i+1) % 1 == 0)
//            {
//                InputFile << endl;
//            }
//        }
//    }
//
//
//    exit(0);

    printf("##########Output Done###########\n");


    /*post process*/

    std::vector<ObjBbox_t> candidate_boxes;
    std::vector<ObjBbox_t> person_list;

    generateBBox(out.data, candidate_boxes, img.w, img.h);

    nms(candidate_boxes, person_list);
    std::vector<ObjBbox_t> person_list_res;

    trans_coords(src, dst, person_list, person_list_res);

    for (int i = 0; i < person_list_res.size(); i++) {
        auto person = person_list_res[i];
        printf("box:   ");
        printf("%3.2f ",person.x0);
        printf("%3.2f ",person.y0);
        printf("%3.2f ",person.x1);
        printf("%3.2f ",person.y1);
        printf("%3.2f ",person.score);
        printf("\n");
    }

    jzdl::net_destory(persondet);

    return 0;
}

void copyMakeBorder(jzdl::Mat<uint8_t> in, jzdl::Mat<uint8_t>& out, int top ,int bottom, int left, int right, int value)
{
    jzdl::Mat<uint8_t> dst(in.w + left + right, in.h + top + bottom, 3);
    dst.fill(value);
    int dst_widthstep = (in.w + left + right) * 3;
    int in_widthstep =  in.w * 3;

    for (int i = 0 ; i < in.h ; i ++ )
    {
        memcpy(dst.data + (top + i) * dst_widthstep + left * 3, in.data + i * in_widthstep, in.w * 3);
    }
    out = dst.clone();
}


void letter_box(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, int value) { 
    int in_w = in.w;
    int in_h = in.h;
    int out_w, out_h;
    int dw, dh;
    float r;
    int top, bottom, left, right;
    //Scale ratio (new / old)
    r = float(det_size)/max(in_w, in_h);
    out_w = int(round(in_w * r));
    out_h = int(round(in_h * r));
    dw = (det_size - out_w) % 32;
    dh = (det_size - out_h) % 32;
    jzdl::Mat<uint8_t> temp(out_w, out_h, 3);
    if(out_w != in_w || out_h != in_h){
        // out.create(out_h, out_w, MXU_8UC3);
        jzdl::resize(in, temp);
//        resize(in, out, out.size(), 0, 0, MXU_INTER_LINEAR );
    } else{      
        temp = in.clone();
    }
    top = int(round(float(dh)/2 - 0.1));
    bottom = int(round(float(dh)/2 + 0.1));
    left = int(round(float(dw)/2 - 0.1));
    right = int(round(float(dw)/2 + 0.1));
//	printf("top:%d bottom:%d left:%d right:%d\n", top, bottom, left, right);
    copyMakeBorder(temp, out, top, bottom, left, right, 128);
}



void trans_coords(const jzdl::Mat<uint8_t>& img, jzdl::Mat<uint8_t>& working_img, vector<ObjBbox_t> in_boxes, vector<ObjBbox_t>& out_boxes){
    int ori_w, ori_h, new_w, new_h;
    float gain;
    float pad_x, pad_y;
    ori_w = img.w;
    ori_h = img.h;
    new_w = working_img.w;
    new_h = working_img.h;
    gain = float(max(new_w, new_h))/float(max(ori_w, ori_h));
    pad_x = (new_w - ori_w * gain)/2;
    pad_y = (new_h - ori_h * gain)/2;
    for(int i = 0; i < in_boxes.size(); i++) {
        ObjBbox_t aa = in_boxes[i];

        ObjBbox_t bb;
        bb.x0 = max(int((aa.x0-pad_x)/gain), 0);
        bb.x1 = min(int((aa.x1-pad_x)/gain), img.w-1);
        bb.y0 = max(int((aa.y0-pad_y)/gain), 0);
        bb.y1 = min(int((aa.y1-pad_y)/gain), img.h-1);
        bb.score = aa.score;
        out_boxes.push_back(bb);
    }
}


void generateBBox(float * p, vector<ObjBbox_t> & candidate_boxes, int img_w, int img_h){

  float person_threshold = 0.3; 
  int classes = 1;
  int box_num = 3;
  float nms_threshold = 0.3;
  vector<float> anchor = {81.0, 82.0, 135.0, 169.0, 344.0, 319.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0};
  vector<float> strides = {32.0,16.0};
  int onechannel = 5 + classes;// 4/location + 1/obj_score + classes 

  for (int s = 0; s < strides.size(); s++){
        int height_ = img_h/strides[s];
        int width_ = img_w/strides[s];
        for(int h=0;h<height_;h++)
        {
            for(int w=0;w<width_;w++)
            {
                for(int n=0;n<box_num;n++)
                {
                    float xptr = p[h*(width_*box_num*onechannel)+w*(box_num*onechannel)+n*onechannel];
                    float yptr = p[h*(width_*box_num*onechannel)+w*(box_num*onechannel)+n*onechannel+1];
                    xptr = 1.f / (1.f + exp(-xptr));
                    yptr = 1.f / (1.f + exp(-yptr));
                    float wptr = p[h*(width_*box_num*onechannel)+w*(box_num*onechannel)+n*onechannel+2];
                    float hptr = p[h*(width_*box_num*onechannel)+w*(box_num*onechannel)+n*onechannel+3];
                    float box_score_ptr = p[h*(width_*box_num*onechannel)+w*(box_num*onechannel)+n*onechannel+4];
                    box_score_ptr = 1.f / (1.f + exp(-box_score_ptr));


                    float anchor_w = anchor[s*box_num*2+n * 2] / img_w;
                    float anchor_h = anchor[s*box_num*2+n * 2 + 1] / img_h;
                    float box_w = exp(wptr) * anchor_w;
                    float box_h = exp(hptr) * anchor_h;


                    float xmin = (xptr + w) / width_ - box_w * 0.5f;
                    float ymin = (yptr + h) / height_ - box_h * 0.5f;
                    float xmax = xmin + box_w;
                    float ymax = ymin + box_h;

                    int class_index = 0;
                    float class_score = 0.f;
                    for (int c =0;c <classes;c++)
                    {
                        float score =  p[h*(width_*box_num*onechannel)+w*(box_num*onechannel)+n*onechannel+5+c];
                        //softmax
                        score = 1.f / (1.f + exp(-score));
                        if (score>class_score)
                        {
                            class_index = c;
                            class_score = score;
                        }
                    }

                    float prob = box_score_ptr;// * class_score;  
                    //if (prob >= 0)
                    if (prob >= person_threshold)
                    {
                        ObjBbox_t object;
                        object.x0 = xmin*img_w;
                        object.y0 = ymin*img_h;
                        object.x1 = xmax*img_w;
                        object.y1 = ymax*img_h;
                        object.score = prob;
                        candidate_boxes.push_back(object);
                    }
                }
            }
        }
        p+= height_*width_*box_num*onechannel;
    }
}
