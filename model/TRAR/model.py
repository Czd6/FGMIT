import torch.nn as nn
import torch
from model.TRAR.trar import FGMIT_ED
from model.TRAR.fc import MLP
from model.TRAR.cls_layer import cls_layer_both
import torch.nn.functional as F

class FGMIT(nn.Module):
    def __init__(self, opt):
        super(FGMIT, self).__init__()
        self.backbone = FGMIT_ED(opt)
        self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        self.att = nn.Linear(opt["hidden_size"], 1, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()
        self.classifier_fuse = nn.Linear(opt["hidden_size"], 2)
        self.classifier_text = nn.Linear(opt["hidden_size"], 2)
        self.classifier_image = nn.Linear(opt["hidden_size"], 2)

    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)

        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
        )

        lang_feat = torch.mean(lang_feat, dim=1)
        img_feat = torch.mean(img_feat, dim=1)


        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat