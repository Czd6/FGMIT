import torch
import timm
import model
from transformers import RobertaModel, ViTModel, AutoImageProcessor
from .MSG_models.msg_transformer import MSGTransformer
from .MSG_models import build_model

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class FGMIT(torch.nn.Module):
    def __init__(self, bertl_text, opt, msg_config):
        super(FGMIT, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.msg_config = msg_config
        if not self.opt["finetune"]:
            freeze_layers(self.bertl_text)
            freeze_layers(self.vit)
        assert ("input1" in opt)
        assert ("input2" in opt)
        assert ("input3" in opt)
        self.input1 = opt["input1"]
        self.input2 = opt["input2"]
        self.input3 = opt["input3"]
        self.MSG = build_model(msg_config)
        self.trar = model.TRAR.FGMIT(opt)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(opt["output_size"], 2)
        )

    def forward(self, input):

        bert_embed_text = self.bertl_text.embeddings(input_ids=input[self.input1])

        for i in range(self.opt["roberta_layer"]):
            bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
            bert_embed_text = bert_text

        img_feat = self.MSG(input[self.input2], bert_embed_text, input[self.input3].unsqueeze(1).unsqueeze(2))

        (score, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text, input[self.input3].unsqueeze(1).unsqueeze(2))


        del bert_embed_text, bert_text, img_feat

        return score, lang_emb, img_emb


def build_FGMIT(opt, msg_config, requirements):
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    return FGMIT(bertl_text, opt, msg_config)
