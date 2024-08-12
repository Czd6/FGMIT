import sys
sys.path.append("../..")
from model.Optimizers import build_Adam
from model.loss_function import build_CrossentropyLoss_ContrastiveLoss, build_BCELoss, build_CrossEntropyLoss, build_CrossEntropyLoss_weighted
import model.TRAR
from model.FGMIT import build_FGMIT
_models={
    "FGMIT":build_FGMIT
}

_optimizers={
    "Adam":build_Adam
}

_loss={
    "CrossEntropyLoss":build_CrossEntropyLoss,
    "BCELoss":build_BCELoss,
    "CrossentropyLoss_ContrastiveLoss": build_CrossentropyLoss_ContrastiveLoss,
    "Crossentropy_Loss_weighted": build_CrossEntropyLoss_weighted
}