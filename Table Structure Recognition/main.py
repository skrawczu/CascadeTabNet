from border import border
from mmdet.apis import inference_detector, show_result, init_detector
import cv2
from Functions.blessFunc import borderless
import lxml.etree as etree
import glob
import os


############ To Do ############
#image_path = '/content/models/MyDrive/anaplan_data/images/test/'
image_path = '/content/models/MyDrive/test_data/'
#xmlPath = '/content/models/MyDrive/examples/anurag_set/'

config_fname = "/content/models/MyDrive/models/struct_model/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
#config_fname = "/content/models/MyDrive/models/top50_model/cascade_mask_rcnn_hrnetv2p_w32_20e.py"

checkpoint_path = "/content/models/MyDrive/models/struct_model/"
#checkpoint_path = "/content/models/MyDrive/models/top50_model/"

epoch = 'struct_model.pth'
#epoch = 'invoice_table_det_model.pth'
##############################


model = init_detector(config_fname, checkpoint_path+epoch)

# List of images in the image_path
#imgs = glob.glob(image_path)
imgs = [image_path + i for i in os.listdir(image_path) if i.endswith('.png')]
print(imgs)
for i in imgs:
    print(i)
    result = inference_detector(model, i)
    res_border = []
    res_bless = []
    res_cell = []
    root = etree.Element("document")
    ## for border
    for r in result[0][0]:
        if r[4]>.7:
            res_border.append(r[:4].astype(int))
    ## for cells
    for r in result[0][1]:
        if r[4]>.7:
            r[4] = r[4]*100
            res_cell.append(r.astype(int))
    ## for borderless
    for r in result[0][2]:
        if r[4]>.7:
            res_bless.append(r[:4].astype(int))

    ## if border tables detected 
    if len(res_border) != 0:
        ## call border script for each table in image
        for res in res_border:
            try:
                root.append(border(res,cv2.imread(i)))  
            except:
                pass
    if len(res_bless) != 0:
        if len(res_cell) != 0:
            for no,res in enumerate(res_bless):
                root.append(borderless(res,cv2.imread(i),res_cell))

    myfile = open(xmlPath+i.split('/')[-1][:-3]+'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True,encoding="unicode"))
    myfile.close()