import pickle
import torch
import json

def load_file(filename):
    txt = []
    f1 = open(filename, "r", encoding='utf-8')
    datas = json.load(f1)
    for data in datas:
        text = data['text']
        txt.append(text)
    return txt

class loader_text:
    def __init__(self):
        self.name="text"
        self.require=["tokenizer_roberta"]

    def prepare(self,input,opt):
        self.text ={
            "train":load_file(opt["data_path"] + "train.json"),
            "test":load_file(opt["data_path"] + "test.json"),
            "valid":load_file(opt["data_path"] + "valid.json")
        }
        if "len" not in opt:
            opt["len"]=100
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        self.tokenizer_roberta=input[list(input.keys())[0]]

        self.text_mask = {
            "train":[],
            "test":[],
            "valid":[]
        }
        self.text_id = {
            "train":[],
            "test":[],
            "valid":[]     
        }
        for mode in self.text.keys():
            for index, text in enumerate(self.text[mode]):
                indexed_tokens_for_text = self.tokenizer_roberta(text)['input_ids']   # 将文本转变成对应的index
                if len(indexed_tokens_for_text) > self.len:  # 如果长度超出限制 则截断
                    indexed_tokens_for_text=indexed_tokens_for_text[0:self.len]
                text_mask=torch.BoolTensor([0]*len(indexed_tokens_for_text)+[1]*(self.len-len(indexed_tokens_for_text)))
                indexed_tokens_for_text+=[self.pad]*(self.len-len(indexed_tokens_for_text))
                text_id = torch.tensor(indexed_tokens_for_text) #将index转换成Tensor类型
                self.text_mask[mode].append(text_mask)
                self.text_id[mode].append(text_id)


    def get(self,result,mode,index):
        result["text_mask"]= self.text_mask[mode][index]
        result["text"]=self.text_id[mode][index]


    def getlength(self,mode):
        return len(self.text[mode])