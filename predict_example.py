
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"

import torch


from MSIGN import MSIGN
from utils import *


import warnings
warnings.filterwarnings('ignore')


# %%

if __name__ == '__main__':
    data_root = './data'
    device = torch.device('cuda')
    valid_dir1 = os.path.join(data_root, 'example/21244')
    pred_dir1 = os.path.join(valid_dir1, 'Atom_Graph2-21244.dgl')


    Mgraph1, label1 = torch.load(pred_dir1)
    Mgraph1 = Mgraph1.to(device)


    model = MSIGN(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
    # fine-tune不冻结
    model.load_state_dict(torch.load("./model/full_finetune_model.pt"))



    model.eval()
    pred_lp1, pred_pl1 = model(Mgraph1)
    pred1 = (pred_lp1 + pred_pl1) / 2



    print("example:Prediction:%.4f, Label:%.4f" % (pred1, label1))



# %%
