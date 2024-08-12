import pickle
import torch
import os
import numpy as np
# from torchvision import transforms
# from torchtoolbox.transform import Cutout
from PIL import Image
import json

def load_file(filename):
    image_id = []
    f1 = open(filename, "r", encoding='utf-8')
    datas = json.load(f1)
    for data in datas:
        id = data['image_id']
        image_id.append(id)
    return image_id

class loader_img:
    def __init__(self):
        self.name="img"
        self.require=[]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file(opt["data_path"] + "train.json"),
            "test":load_file(opt["data_path"] + "test.json"),
            "valid":load_file(opt["data_path"] + "valid.json")
        }
        self.transform_image_path = opt["transform_image"]

    def get(self,result,mode,index):
        img_path=os.path.join(
                self.transform_image_path,
                "{}.npy".format(self.id[mode][index])
            )
        img = torch.from_numpy(np.load(img_path))
        # img = np.load(img_path)
        # if img.shape[0]==3:
        #     img = (np.transpose(img, (1,2,0))+1)/2.0*255.0
        # elif img.shape[0]==1:
        #     img = (img[0] + 1) / 2.0 * 255.0

        # img = Image.fromarray(np.uint8(img))
        # preprocess = transforms.Compose([
        #     transforms.CenterCrop(196),
        #     Cutout(0.5),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(0.5,0.5,0.5,0.3),
        #     transforms.Pad(14,fill=(255,0,0)),
        #     transforms.ToTensor()
        # ])

        # img = preprocess(img=img)
        # img = torch.Tensor(img)
        result["img"]=img
    

    def getlength(self,mode):
        return len(self.id[mode])
    