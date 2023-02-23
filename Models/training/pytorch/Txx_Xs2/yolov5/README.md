1.Version Requirments
		  python
		  pytorch	  
		  cuda
		  cudnn
		  gcc
Please note that the above versions are consistent with the ones provided in the release of the plug-in. If there are any changes, please inform Ingenic's technical staff in time to update the plug-in.
The rest of the dependencies can be installed as needed by referring to requirements.txt or the actual running prompt.

2.Data Preparation
	   Take COCO_2017 as an example, the COCO2017 dataset is downloaded and extracted, and the annotation file is in .json format
	   Download Address:
	   Image dataset：
	   		 http://images.cocodataset.org/zips/train2017.zip
			 http://images.cocodataset.org/zips/test2017.zip
			 http://images.cocodataset.org/zips/val2017.zip
	   Image annotation:
	   		 http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
			 http://images.cocodataset.org/annotations/image_info_test2017.zip
			 http://images.cocodataset.org/annotations/annotations_trainval2017.zip

The script to generate the data is in the COCO_forYOLO folder.
	  (1)Run python batch_split_annotation_foryolo.py (note that the coco path 'coco_data_dir=' is modified in the program).
  	  (2)After running, three folders will be created under coco path: person/data/images, person/data/ImageSets, person/data/labels; put train2017.txt, val2017.txt, test2017.txt under ImageSets into persondet/data/coco folder, and store the absolute path of the images in the txt file.
	
3.Training
	 (1)sh multi_gpu_train.sh/sh singlel_gpu_train.sh
     (2)You can use kmeans_anchor.py to get the anchor corresponding to your data and modify the anchor parameter in cfg/persondet.cfg, or use the original anchor parameter;
     (3)The training configuration parameters can be found in models/commom.py, such as is_quantize, bitw, and bita:
       32bit, Set:is_quantize = 0, bita = 32, see specific code for the rest
       8bit, Set:is_quantize = 1, bita = 8, see specific code for the rest
       4bit, Set:is_quantize = 1, bita = 4, see specific code for the rest
	   
    In addition, floating point training without pre-training model, --weight for '', 8bit training when loading the obtained precision enough floating point model, while appropriate to reduce the learning rate; and so on, 4bit loading 8bit, if the effect can also be based on this try w2a4, so step by step to advance, the effect is more.
	
	 (4)train.py --project can be set to save the model path, under runs/train/project, there are two models saved, best.pt the best trained so far and last.pt the latest trained so far, the test results are saved in result.txt, which can be viewed at any time.

4.Test models and test images
	   #ATTENTION:When testing, be sure to pay attention to the bit settings in the common and test model correspondence, otherwise the results will be inaccurate
	   
	   By python3 detect.py -h view to select the required parameters, by setting source can detect images and videos, detection results can be set to display (--view-img) or save (--save-img)
- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`

The example is as follows, if you need other operations you can select the corresponding parameters;
sh detect.sh
Note: The model configuration for detection must be the same as the training configuration (bitwidth)

Test model accuracy:
sh test.sh

The model to be tested is specified by ---weights, the validation set is determined by val2017.txt in data/coco-person.yaml, and the rest of the parameters are given according to the actual needs

5.Model conversion(*.onnx)
 Generate .onnx files：sh convert_onnx.sh
    input：Specify the path of the model to be transferred.
    output：.onnx, location and .pt in the same directory.

6. Enjoy it.