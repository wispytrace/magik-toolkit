#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include "net.h"
#include "img_input.h"
#include "fstream"
#include <iomanip>
#include "utils.h"
#include <sys/time.h>

using namespace std;
using namespace jzdl;

#define det_size 416

// #define MEM_MODEL
#ifdef MEM_MODEL
#include "magik_model_persondet.mk.h"
#endif

static const int max_wh = 4096, min_wh = 2;
static const std::vector<int> strides = {32, 16};
static const std::map<int, std::vector<float> > anchors = {{32, {81.f,82.f, 135.f,169.f, 344.f,319.f}}, {16, {23.f,27.f, 37.f,58.f, 81.f,82.f}}};
static const int num_box_per_grid = 3;
static const float nms_threshold = 0.3f;
static const int num_class = 1; // person

void generateBBox(float * p, vector<ObjBbox_t> & candidate_boxes, int img_w, int img_h);
void copyMakeBorder(jzdl::Mat<uint8_t> in, jzdl::Mat<uint8_t>& out, int top ,int bottom, int left, int right, int value);
void letter_box(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, int value);
void trans_coords(const jzdl::Mat<uint8_t>& img, jzdl::Mat<uint8_t>& working_img, vector<ObjBbox_t> in_boxes, vector<ObjBbox_t>& out_boxes);
void network_preprocess(jzdl::Mat<uint8_t>& img, jzdl::Mat<float>& out);
void post_preprocess(jzdl::Mat<uint8_t>& img, jzdl::Mat<float>& out, std::vector<ObjBbox_t>& box);

int main(int argc, char* argv[]) {
    /* network feature process */
    printf("imagedata size:%d\n", sizeof(image));
    jzdl::Mat<uint8_t> src(640, 480, 3, (uint8_t*)image); // image is bgr ptr
    jzdl::Mat<float> out;
    network_preprocess(src, out);
    
    /*post process*/
    std::vector<ObjBbox_t> person_list_res;
    post_preprocess(src, out, person_list_res);
    
	/* print box result */
    for (unsigned int i = 0; i < person_list_res.size(); i++) {
        auto person = person_list_res[i];
        printf("box:   ");
        printf("%f ",person.x0);
        printf("%f ",person.y0);
        printf("%f ",person.x1);
        printf("%f ",person.y1);
        printf("%f ",person.score);
        printf("\n");
    }
    return 0;
}

static float __max(float a, float b) {
    float x = a > b ? a : b;
    return x;
}

static float __min(float a, float b) {
    float x = a < b ? a : b;
    return x;
}

static inline float _line_intersection(float start1, float end1, float start2, float end2) {
    if(start1 >= end1 || start2 >= end2) return 0.0f;
    float max_start = __max(start1, start2);
    float min_end   = __min(end1, end2);
    return min_end > max_start ? min_end - max_start : 0.0f;
}


inline static float box_area(const ObjBbox_t& box) {
    return (box.x1 - box.x0) * (box.y1 - box.y0);
}

static float iou(const ObjBbox_t& a, const ObjBbox_t& b) {
    float h_inter_section = _line_intersection(a.y0, a.y1, b.y0, b.y1);
    float w_inter_section = _line_intersection(a.x0, a.x1, b.x0, b.x1);
    float inter_area = h_inter_section * w_inter_section;
    float area_a = box_area(a);
    float area_b = box_area(b);
    if (area_a <= 0 || area_b <= 0) return 0.f;
    return inter_area / (area_a + area_b - inter_area);
}

static std::vector<float> box_iou(const std::vector<ObjBbox_t>& all, const std::vector<ObjBbox_t>& selected) {
    std::vector<float> inters;
    for (auto it : selected) {
        for (auto _it : all) {
            float __iou = iou(it, _it);
            if (__iou > 0.3f) {
                inters.push_back(__iou);
            } else {
                inters.push_back(0.f);
            }
        }
    }
    return inters;
}

static void to_weights(const std::vector<ObjBbox_t>& all, std::vector<float>& inters) {
    std::vector<float> ws;
    ws.resize(inters.size()/all.size(), 0);
    for (unsigned int n = 0; n < inters.size()/all.size(); ++n) {
        int stride = n*all.size();
        for (int i = 0; i < (int)all.size(); ++i) {
            inters[i + stride] *= all[i].score;
            ws[n] += inters[i+stride];
        }
    }
    
    for (unsigned int n = 0; n < inters.size()/all.size(); ++n) {
        int stride = n*all.size();
        for (int i = 0; i < (int)all.size(); ++i) {   
            inters[i + stride] /= ws[n];
        }
    }
}

void __mm(const std::vector<float>& weight, const std::vector<ObjBbox_t>& all, std::vector<ObjBbox_t>& psersons) {
    if (weight.size() != psersons.size()*all.size()) {
        std::cout<<"Wrong weight size"<<std::endl;
        abort();
    }
    for (int i = 0; i < (int)psersons.size(); ++i) {
        float x0 = 0;
        float y0 = 0;
        float x1 = 0;
        float y1 = 0;
        for (int __i = 0; __i < (int)all.size(); ++__i) {
            x0 += (weight[i*all.size() + __i] * all[__i].x0);
            y0 += (weight[i*all.size() + __i] * all[__i].y0);
            x1 += (weight[i*all.size() + __i] * all[__i].x1);
            y1 += (weight[i*all.size() + __i] * all[__i].y1);
        }
        psersons[i].x0 = x0;
        psersons[i].y0 = y0;
        psersons[i].x1 = x1;
        psersons[i].y1 = y1;
    }
}

void post_preprocess(jzdl::Mat<uint8_t>& src, jzdl::Mat<float>& out, std::vector<ObjBbox_t>& box) {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    double time_last = tv.tv_sec*1000000 + tv.tv_usec;

	jzdl::Mat<uint8_t> dst;
    letter_box(src, dst, 128);
    std::vector<ObjBbox_t> candidate_boxes;
    std::vector<ObjBbox_t> person_list;
    generateBBox(out.data, candidate_boxes, dst.w, dst.h);
    
    nms(candidate_boxes, person_list);
    std::vector<float> inters = box_iou(candidate_boxes, person_list);
    to_weights(candidate_boxes, inters);
    __mm(inters, candidate_boxes, person_list);
    trans_coords(src, dst, person_list, box);

    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    float time_ms = time_last*1.0/1000;
    printf("runing time post_preprocess: %f ms\n", time_ms);

}


void network_preprocess(jzdl::Mat<uint8_t>& src, jzdl::Mat<float>& out)
{
    jzdl::BaseNet *persondet = jzdl::net_create();

#ifdef MEM_MODEL
    persondet->load_model((const char*)magik_model_persondet_mk, true);
#else
    std::string model_file_path = "magik_model_persondet.bin";
    persondet->load_model(model_file_path.c_str());
#endif

    std::vector<uint32_t> input_shape = persondet->get_input_shape();
    int input_index = persondet->get_model_input_index();
    int output_index = persondet->get_model_output_index();
    printf("input_shape.w=%d, input_shape.h=%d, input_shape.c=%d\n", input_shape[0], input_shape[1], input_shape[2]);
    printf("input_index=%d, output_index=%d\n", input_index, output_index);

    jzdl::Mat<uint8_t> dst;
    printf("%d %d %d\n", src.data[0], src.data[1], src.data[2]);

    letter_box(src, dst, 128);
    printf("%d %d %d\n", src.w, src.h, src.c);

    image_sub(dst, 128);
    jzdl::Mat<int8_t> img(dst.w, dst.h, dst.c, (int8_t*)dst.data);
    printf("%d %d %d\n", dst.w, dst.h, dst.c);
    printf("%d %d %d\n", dst.data[100], dst.data[200], dst.data[800]);
    struct timeval tv;

    gettimeofday(&tv, NULL);
    double time_last = tv.tv_sec*1000000 + tv.tv_usec;

    persondet->input(img);
    persondet->run(out);

    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    float time_ms = time_last*1.0/1000;
    printf("runing time network: %f ms\n", time_ms);

    printf("########Network run end!#######\n");
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
    } else {      
        temp = in.clone();
    }
    top = int(round(float(dh)/2 - 0.1));
    bottom = int(round(float(dh)/2 + 0.1));
    left = int(round(float(dw)/2 - 0.1));
    right = int(round(float(dw)/2 + 0.1));
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
    for(unsigned int i = 0; i < in_boxes.size(); i++) {
        ObjBbox_t aa = in_boxes[i];
        ObjBbox_t bb;
        bb.x0 = round(__max((aa.x0-pad_x)/gain, 0));
        bb.x1 = round(__min((aa.x1-pad_x)/gain, ori_w));
        bb.y0 = round(__max((aa.y0-pad_y)/gain, 0));
        bb.y1 = round(__min((aa.y1-pad_y)/gain, ori_h));
        bb.score = aa.score;
        out_boxes.push_back(bb);
    }
}

static float __sigmoid(float x) {
    return 1 / (1+ exp(-x));
}

static bool refine_wh(float w, float h) {
    return w < max_wh && h < max_wh && w > min_wh && h > min_wh;
}

static void xywh2xyxy(float& x, float& y, float& w, float& h) {
    float __x0 = x - w/2;
    float __y0 = y - h/2;
    float __x1 = x + w/2;
    float __y1 = y + h/2;
    x = __x0;
    y = __y0;
    w = __x1;
    h = __y1;
}

void generateBBox(float *p, vector<ObjBbox_t> &candidate_boxes, int img_w, int img_h) {
    for (int s : strides) {
        int grid_w = img_w / s;
        int grid_h = img_h / s;
        int s_stride = 5+num_class;
        int w_stride = num_box_per_grid*s_stride;
        int h_stride = grid_w*w_stride;
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                for (int n = 0; n < num_box_per_grid; ++n) {
                    int index = h*h_stride + w*w_stride + n*s_stride;
                    float cx = (__sigmoid(p[index + 0]) + w) * s;
                    float cy = (__sigmoid(p[index + 1]) + h) * s;
                    float anchor_w = anchors.at(s)[n*2];
                    float anchor_h = anchors.at(s)[n*2+1];
                    float w = exp(p[index + 2]) * anchor_w;
                    float h = exp(p[index + 3]) * anchor_h;
                    float obj_score = __sigmoid(p[index + 4]);
                    float cls_score = __sigmoid(p[index + 5]);
                    float score = obj_score*cls_score;
                    xywh2xyxy(cx, cy, w, h); // center x y and wh to x0 y0 x1 y1
                    if (0.3f < obj_score && refine_wh(w, h)) {
                        ObjBbox_t person;
                        person.x0 = cx;
                        person.y0 = cy;
                        person.x1 = w;
                        person.y1 = h;
                        person.score = score;
                        candidate_boxes.push_back(person);
                    }
                }
            }
        }
        p += (img_w / strides[0]) * (img_h / strides[0]) * w_stride;
    }
}
