import os
import subprocess
import sys
import shutil

CURDIR = os.path.dirname(os.path.realpath(__file__))
redo = True

coco_data_dir = "/data/coco_2017" #your coco dataset path,  you need to change it

anno_sets = ["instances_train2017", "instances_val2017", "image_info_test2017"]
anno_dir = "{}/annotations".format(coco_data_dir)
out_anno_dir = "{}/person/data/labels".format(coco_data_dir)
out_img_dir = "{}/person/data/images".format(coco_data_dir)
imgset_dir = "{}/person/data/ImageSets".format(coco_data_dir)

####creat labels
for i in range(0, len(anno_sets)):
    anno_set = anno_sets[i]
    anno_file = "{}/{}.json".format(anno_dir, anno_set)
    if not os.path.exists(anno_file):
        print("{} does not exist".format(anno_file))
        continue
    anno_name = anno_set.split("_")[-1]
    out_dir = out_anno_dir
    imgset_file = "{}/{}.txt".format(imgset_dir, anno_name)
    print(out_dir, imgset_file, anno_file)
    if redo or not os.path.exists(out_dir):
        cmd = "python {}/split_annotation_foryolo.py --out-dir={} --imgset-file={} {}" \
                .format(CURDIR, out_dir, imgset_file, anno_file)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]


####creat txt
for root,dirs,files in os.walk(imgset_dir):
    for name in files:
        input_txt = os.path.join(root,name)
        out_txt = os.path.join(imgset_dir,name)
        with open(input_txt,'r') as f:
            lines = f.readlines()
        ff = open(out_txt,'w')
        for line in lines:
            ff.write(out_img_dir+'/'+line.strip()+'.jpg\n')
        ff.close()


####copy images
train_image_subsets = os.path.join(coco_data_dir,'train2017')
val_image_subsets = os.path.join(coco_data_dir,'val2017')
test_image_subsets = os.path.join(coco_data_dir,'test2017')

if not os.path.exists(out_img_dir):
    os.makedirs(out_img_dir)

def change(path, path1):
    for f in os.listdir(path):
        if os.path.isfile(path + os.path.sep + f):
            a, b = os.path.splitext(f)
            if b != '.py':
                shutil.copy(path + os.sep + f, path1)
        elif os.path.isdir(path + os.path.sep + f):
            change(path + os.sep + f, path1)

# Copy annotations from subset to labels.
change(train_image_subsets, out_img_dir)
change(val_image_subsets, out_img_dir)
change(test_image_subsets, out_img_dir)
