import pickle
import json

def load_file(filename):
    label = []
    f1 = open(filename, "r", encoding='utf-8')
    datas = json.load(f1)
    for data in datas:
        lab = data['label']
        label.append(lab)
    return label

class loader_label:
    def __init__(self):
        self.name="label"
        self.require=[]

    def prepare(self,input,opt):
        if opt["test_label"]:
            self.label ={
                "train":load_file(opt["data_path"] + "train.json"),
                "test":load_file(opt["data_path"] + "test.json"),
                "valid":load_file(opt["data_path"] + "valid.json")
            }
        else:
            self.label ={
                "train":load_file(opt["data_path"] + "train_labels"),
                "valid":load_file(opt["data_path"] + "valid_labels")
            }

    def get(self,result,mode,index):
        result["label"]=self.label[mode][index]
        # result["index"]=index

    def getlength(self,mode):
        return len(self.label[mode])