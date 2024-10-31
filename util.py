import torch
import dgl
import numpy as np

def collate_graphs(batch):
    gs, g_augs, ys = map(list, zip(*batch))
    
    
    gs = dgl.batch(gs)

    g_augs = list(zip(*g_augs))
    g_augs = [dgl.batch(g_aug) for g_aug in g_augs]

    if len(ys[0]) == 1:
        ys = torch.Tensor(np.hstack(ys))
    
    elif len(ys[0]) > 1:
        ys = torch.Tensor(np.vstack(ys))

    return gs, g_augs, ys
