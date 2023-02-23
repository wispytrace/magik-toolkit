#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include "net.h"
#include <fstream>
#include <iomanip>
//#include "iaac.h"
#include "utils.h"
#include "img_input.h"

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#define MEM_MODEL
#ifdef MEM_MODEL
#include "magik_model_facedet.mk.h"
#endif

using namespace std;
using namespace jzdl;
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

typedef struct objbox {
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
	int clsid;	
} objbox; 

void trans_coords(int ori_w, int ori_h, int input_w, int input_h, vector<objbox> &in_boxes) {
	float scale;
    float pad_x, pad_y;
    scale = min((float(input_w)/ori_w), (float(input_h)/ori_h));
    pad_x = (input_w - ori_w * scale) / 2;
    pad_y = (input_h - ori_h * scale) / 2;
    for(int i = 0; i < in_boxes.size(); ++i) {
        in_boxes[i].x0 = (in_boxes[i].x0 - pad_x) / scale;
		in_boxes[i].x1 = (in_boxes[i].x1 - pad_x) / scale;
        in_boxes[i].y0 = (in_boxes[i].y0 - pad_y) / scale;
        in_boxes[i].y1 = (in_boxes[i].y1 - pad_y) / scale;
    }
}

void prepare_data_resize_pad(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, 
                             int model_in_w, int model_in_h) {
	
	jzdl::PadInfo_t padinfo;
    int input_w = in.w;
    int input_h = in.h;
	
    int output_w, output_h;
    int dw, dh;
    int top, bottom, left, right;
	
    int con_w = model_in_w;
    int con_h = model_in_h;

    float h_scale = float(con_h) / float(input_h);
    float w_scale = float(con_w) / float(input_w);

    float scale = min(h_scale, w_scale);
    output_w = int(round(input_w * scale));
    output_h = int(round(input_h * scale));
    dw = (con_w - output_w);
    dh = (con_h - output_h);
    
    jzdl::Mat<uint8_t> temp(output_w, output_h, 3);

    if(output_h != input_h || output_h != input_w) {
		jzdl::resize(in, temp);

    } else{
        temp = in.clone();
    }

    top = int(round(float(dh)/2 - 0.1));
    bottom = int(round(float(dh)/2 + 0.1));
    left = int(round(float(dw)/2 - 0.1));
    right = int(round(float(dw)/2 + 0.1));
	
	
	padinfo.top = top;
	padinfo.bottom = bottom;
	padinfo.left = left;
	padinfo.right = right;
	padinfo.type = PAD_CONSTANT;
	padinfo.value = 128;
	jzdl::image_pad(temp, out, padinfo);//填充灰色的值(128)
    return ;
}

void prepare_data_pad_resize(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, 
                             int model_in_w, int model_in_h) {
	
	jzdl::PadInfo_t padinfo;
    int input_w = in.w;
    int input_h = in.h;
	
    int output_w, output_h;
    int dw, dh;
    int top, bottom, left, right;
	
    int con_w = model_in_w;
    int con_h = model_in_h;

	float h_scale = float(input_h) / float(con_h);
    float w_scale = float(input_w) / float(con_w);

    float scale = max(h_scale, w_scale);
    output_w = int(round(con_w * scale));
    output_h = int(round(con_h * scale));
    dw = (output_w - input_w);
    dh = (output_h - input_h);
    jzdl::Mat<uint8_t> temp(output_w, output_h, 3);
	
	top = int(round(float(dh)/2 - 0.1));
    bottom = int(round(float(dh)/2 + 0.1));
    left = int(round(float(dw)/2 - 0.1));
    right = int(round(float(dw)/2 + 0.1));
	
	padinfo.top = top;
	padinfo.bottom = bottom;
	padinfo.left = left;
	padinfo.right = right;
	padinfo.type = PAD_CONSTANT;
	padinfo.value = 128;
	jzdl::image_pad(in, temp, padinfo);//填充灰色的值(128)
	jzdl::resize(temp, out);
    return ;
}


float softmax(std::vector<float>& scores, int anc_indx, int class_indx, int class_numbers) {
    float max = 0.0;
    float sum = 0.0;

    for (int i = 0; i<class_numbers; ++i) {
        if(max < scores[anc_indx*class_numbers+i]) {
            max = scores[anc_indx*class_numbers+i];
        }
    }

  float temp_score = 0.;
  for (int j = 0; j < class_numbers; j++) {
	  float temp_value = exp(scores[anc_indx*class_numbers+j] - max);
      sum += temp_value;
	  if (j == class_indx)
		  temp_score = temp_value;
  }
  
  return temp_score / sum;
}

void generateBBox(std::vector<objbox> &face_list, std::vector<float>& scores, std::vector<std::vector<float>> &priors,
                  std::vector<float> boxes, float score_threshold, int num_anchors, int class_num,int in_w,int in_h) {
    std::vector<ObjBbox_t> gt_bboxes;
    std::vector<ObjBbox_t> bbox_collection;
    for (int clsn = 1; clsn < class_num; ++clsn) {

        for (int i = 0; i < num_anchors; ++i) {
            float temp_score = softmax(scores, i, clsn, class_num);
            if ( temp_score > score_threshold) {
                ObjBbox_t rects;
                float x_center = boxes[i * 4] * 0.1 * priors[i][2] + priors[i][0];
                float y_center = boxes[i * 4 + 1] * 0.1 * priors[i][3] + priors[i][1];
                float w = exp(boxes[i * 4 + 2] * 0.2) * priors[i][2];
                float h = exp(boxes[i * 4 + 3] * 0.2) * priors[i][3];

                rects.x0 = clip(x_center - w / 2.0, 1) * in_w;
                rects.y0 = clip(y_center - h / 2.0, 1) * in_h;
                rects.x1 = clip(x_center + w / 2.0, 1) * in_w;
                rects.y1 = clip(y_center + h / 2.0, 1) * in_h;
                rects.score = clip(softmax(scores,i,clsn,class_num), 1);
                bbox_collection.push_back(rects);
            }
        }

        nms(bbox_collection, gt_bboxes);
	
        bbox_collection.clear();
	
        for (int i = 0; i < gt_bboxes.size(); i++) {
            objbox tmp_objbox;
            auto face = gt_bboxes[i];
            tmp_objbox.x0 = face.x0;
            tmp_objbox.x1 = face.x1;
            tmp_objbox.y0 = face.y0;
            tmp_objbox.y1 = face.y1;
            tmp_objbox.score = face.score;
            tmp_objbox.clsid = clsn;
            face_list.push_back(tmp_objbox);
        }
        gt_bboxes.clear();
    }
}


int main(int argc, char* argv[]) {
//	printf("##############################IAAC  Start#######################################\n");
//    /* iaac */
//    static IAACInfo ainfo = {
//        .license_path = (char*)"/mnt/license.txt",
//        .cid = 1,   // ingenic
//        .fid = 1825793026,  // sentinel_dcr
//        .sn = (char*)"ae7117082a18d846e4ea433bcf7e3d2b",
//    };
//
//    int ret = IAAC_Init(&ainfo);
//    if (ret) {
//            printf("%s:%d -> IAAC_Init error!\n", __func__, __LINE__);
//            return -1;
//    }
//    printf("##############################IAAC  Init#######################################\n");

    printf("imagedata size:%d\n", sizeof(image));
	
    jzdl::BaseNet *facedet = jzdl::net_create();


#ifdef MEM_MODEL
    facedet->load_model((const char*)magik_model_facedet_mk, true);
#else
    std::string model_file_path = "magik_model_facedet.bin";
    facedet->load_model(model_file_path.c_str());
#endif

    std::vector<uint32_t> input_shape = facedet->get_input_shape();
    int input_index = facedet->get_model_input_index();
    int output_index = facedet->get_model_output_index();
    printf("input_shape.w=%d, input_shape.h=%d, input_shape.c=%d\n", input_shape[0], input_shape[1], input_shape[2]);
    printf("input_index=%d, output_index=%d\n", input_index, output_index);

	int sw = 1024;
    int sh = 769;

	int in_w = 320, in_h = 240;
    std::vector<int> w_h_list = {320, 240};
	
    jzdl::Mat<uint8_t> src(sw, sh, 3, (uint8_t*)image);
    jzdl::Mat<uint8_t> dst(in_w, in_h, 3);
	printf("%d,%d,%d\n",src.w,src.h,src.c);
	printf("%d,%d,%d\n",dst.w,dst.h,dst.c);
	//padding
	prepare_data_pad_resize(src,dst,in_w,in_h);
	
   // 	for (int i = 0; i < dst.h * dst.w * dst.c; i++)
   // {
   // 	   printf("%d\n",dst.data[i]);
   // }
   // 	return 0;

    //jzdl::resize(src, dst);

    image_sub(dst, 128);
    jzdl::Mat<int8_t> img(input_shape[0], input_shape[1], input_shape[2], (int8_t*)dst.data);
    jzdl::Mat<float> out;

    printf("########Input Done#######\n");

#ifdef TIME
    struct timeval tv; 
    gettimeofday(&tv, NULL);
    uint64_t last_time = tv.tv_sec * 1000000 + tv.tv_usec;
#endif

    facedet->input(img);
    facedet->run(out);

#ifdef TIME
    gettimeofday(&tv, NULL);
    uint64_t new_time = tv.tv_sec * 1000000 + tv.tv_usec;
    double time_used = (new_time - last_time) * 1.0 / 1000.0;
    printf("time network run used:%fms\n", time_used);
#endif

   // printf("%d,%d,%d\n",out.h,out.w,out.c);
   // ofstream InputFile("img_feature_jzdl.h");
   // for (int i = 0; i < out.h * out.w * out.c; ++i) {
   //     float temp = static_cast<float>(out.data[i]);

   //     InputFile  << temp;
   //     if ((i+1) % 1 == 0) {
   //         InputFile << endl;
   //     }
   // }

#ifdef TIME
    gettimeofday(&tv, NULL);
    last_time = tv.tv_sec * 1000000 + tv.tv_usec;
#endif

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
        {10.0f,  16.0f,  24.0f},
        {32.0f,  48.0f},
        {64.0f,  96.0f},
        {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<std::vector<float>> priors = {};

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }
    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }

    int class_num = 2;
    std::vector<float> scores;
    std::vector<float> boxes;
    int out_size = (int)(out.w*out.h*out.c);
    int out_num = (int)out.data[out.w*out.h*out.c-1];
	float *sc_ptr;
    sc_ptr = out.data;
    float* bb_ptr;

    int data_stide = 0;
    /* generate prior anchors */
    for (int index = 0; index < 4; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }

    for (int i = 0; i<min_boxes.size(); ++i) {
        int width_ = featuremap_size[0][i]; int height_=featuremap_size[1][i];
        int num_anchor = min_boxes[i].size();
		bb_ptr = sc_ptr+width_*height_*num_anchor*class_num;
        for(int h = 0; h < height_; ++h) {
            for(int w = 0; w < width_; ++w) {
                for(int n = 0; n < num_anchor; ++n) {
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4]);
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4+1]);
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4+2]);
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4+3]);

                     for(int cls = 0; cls < class_num; ++cls) {
                       scores.push_back(sc_ptr[h*(width_*num_anchor*class_num)+w*(num_anchor*class_num)+n*class_num+cls]);
                     }
                }
            }
        }
        sc_ptr=bb_ptr+width_*height_*num_anchor*4;
    }

    std::vector<objbox> face_list;
    int num_anchors = priors.size();
	
    generateBBox(face_list, scores, priors, boxes, 0.4, num_anchors, class_num,in_w,in_h);
	trans_coords(sw,sh,in_w,in_h,face_list);

#ifdef TIME
    gettimeofday(&tv, NULL);
    new_time = tv.tv_sec * 1000000 + tv.tv_usec;
    time_used = (new_time - last_time) * 1.0 / 1000.0;
    printf("post_preprocess time used:%fms\n", time_used);
#endif

	for (int i = 0; i < face_list.size(); i++) {
        auto face = face_list[i];
        printf("box:   ");
        printf("%3.2f ",face.x0);
        printf("%3.2f ",face.y0);
        printf("%3.2f ",face.x1);
        printf("%3.2f ",face.y1);
        printf("%3.2f ",face.score);
		printf("%d ",face.clsid);
        printf("\n");
    }
    printf("face_list:%d\n", face_list.size());
    jzdl::net_destory(facedet);
    return 0;
}
