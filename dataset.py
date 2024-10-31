import os, random
import numpy as np
from copy import deepcopy
from dgl.convert import graph
import torch
from torch.distributions import Bernoulli



class Dataset():

    def __init__(self, d_path, d_name, aug_mode, num_views=1):
        self.d_path = d_path
        self.d_name = d_name
        self.aug_mode = aug_mode
        self.num_views = num_views
        self.load()

    def load(self):

        [mol_dict] = np.load(os.path.join(self.d_path,'%s_graph.npz'%self.d_name), allow_pickle=True)
        
        self.mols = mol_dict['mols']
        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.node_attr = np.hstack([self.node_attr, np.zeros(len(self.node_attr)).reshape(-1,1)])

        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
        self.smiles = mol_dict['smiles']
        self.y = mol_dict['y']
        self.task_type = mol_dict['task_type']

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
        self.num_tasks = self.y.shape[1]

    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        y = self.y[idx].astype(float)

        if self.aug_mode == 'none':
            g_aug = [deepcopy(g) for _ in range(self.num_views)]
        
        elif self.aug_mode == 'randaug':
                
            g_aug = []
            
            for _ in range(self.num_views):

                p = random.uniform(0.,0.1)
                random_num = random.sample(list(range(2)), 1)[0]

                if random_num == 0:
                    g_aug_ = deepcopy(g)

                    dist = Bernoulli(p)
                    samples = dist.sample(torch.Size([g_aug_.number_of_nodes()]))
                    mask_nodes = g_aug_.nodes()[samples.bool()]
                
                    if len(mask_nodes) < g_aug_.number_of_nodes():
                        for atom_idx in mask_nodes:
                            g_aug_.ndata['node_attr'][atom_idx,:] = torch.tensor([0 for _ in range(self.node_attr.shape[1]-1)]+[1])

                    g_aug.append(g_aug_)


                elif random_num == 1:
                    g_aug_ = deepcopy(g)
                
                    dist = Bernoulli(p)
                    samples = dist.sample(torch.Size([g_aug_.number_of_nodes()]))
                    deleted_nodes = g_aug_.nodes()[samples.bool()]

                    if len(deleted_nodes) < g_aug_.number_of_nodes():
                        g_aug_.remove_nodes(deleted_nodes)

                    g_aug.append(g_aug_)
        
        return g, g_aug, y

    def __len__(self):
        return self.y.shape[0]
    

