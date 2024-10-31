import os, csv, argparse, random
import numpy as np

from dgl.dataloading import GraphDataLoader
from dgl.data.utils import Subset
from dgllife.utils import ScaffoldSplitter

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch

from trn_cons_reg import Cons_Reg_Trainer
from dataset import Dataset
from util import collate_graphs
from gnn import GINPredictor


def main(args):

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('-- CONFIGURATIONS')
    print(f'--- d_name: {args.d_name}, split: {args.split}, cvid: {args.cvid}, backbone: {args.backbone}, seed: {args.seed}')

    if not os.path.exists('./model/'): os.makedirs('./model/')

    model_path = './model/gnn_%s_%d_%d_%d_%d'%(args.d_name, args.seed, args.split, args.cvid, args.flag_s)

    dataset_1 = Dataset(args.d_path, args.d_name, 'none')
    dataset_2 = Dataset(args.d_path, args.d_name, 'randaug', args.num_views)

    
    if args.split ==0:
        # 100% training dataset
        folds = ScaffoldSplitter.k_fold_split(dataset_1, k=5, scaffold_func = 'smiles', log_every_n = None)
        
        trnval_dataset, tst_dataset = folds[args.cvid]
        trnval_idx, tst_idx = np.array(trnval_dataset.indices), np.array(tst_dataset.indices)
        trn_idx, val_idx = train_test_split(trnval_idx, test_size=0.20, random_state=args.seed)
    
    elif args.split == 1:
        # 50% training dataset
        folds = ScaffoldSplitter.k_fold_split(dataset_1, k=5, scaffold_func = 'smiles', log_every_n = None)

        trnval_dataset, tst_dataset = folds[args.cvid]
        trnval_idx, tst_idx = np.array(trnval_dataset.indices), np.array(tst_dataset.indices)
        _, trnval_idx = train_test_split(trnval_idx, test_size=0.50, random_state=args.seed)
        trn_idx, val_idx = train_test_split(trnval_idx, test_size=0.20, random_state=args.seed)
   
        

    elif args.split == 2:
        # 20% training dataset
        folds = ScaffoldSplitter.k_fold_split(dataset_1, k=5, scaffold_func = 'smiles', log_every_n = None)

        trnval_dataset, tst_dataset = folds[args.cvid]
        trnval_idx, tst_idx = np.array(trnval_dataset.indices), np.array(tst_dataset.indices)
        _, trnval_idx = train_test_split(trnval_idx, test_size=0.20, random_state=args.seed)
        trn_idx, val_idx = train_test_split(trnval_idx, test_size=0.20, random_state=args.seed)
    
    elif args.split == 3:
        # 10% training dataset
        folds = ScaffoldSplitter.k_fold_split(dataset_1, k=5, scaffold_func = 'smiles', log_every_n = None)

        trnval_dataset, tst_dataset = folds[args.cvid]
        trnval_idx, tst_idx = np.array(trnval_dataset.indices), np.array(tst_dataset.indices)       
        _, trnval_idx = train_test_split(trnval_idx, test_size=0.10, random_state=args.seed)
        trn_idx, val_idx = train_test_split(trnval_idx, test_size=0.20, random_state=args.seed)

    print('trn:', len(trn_idx))
    print('val:', len(val_idx))
    print('tst:', len(tst_idx))

    trn_dataset = Subset(dataset_2, trn_idx)
    val_dataset = Subset(dataset_1, val_idx)
    tst_dataset = Subset(dataset_1, tst_idx)

    print('trn aug:', trn_dataset.dataset.aug_mode)
    print('val aug:', val_dataset.dataset.aug_mode)
    print('tst aug:', tst_dataset.dataset.aug_mode)

    trn_loader = GraphDataLoader(dataset = trn_dataset, batch_size = min([128, len(trn_dataset)]), shuffle = True, collate_fn = collate_graphs, drop_last = True)
    val_loader = GraphDataLoader(dataset = val_dataset, batch_size = min([128, len(val_dataset)]), shuffle = False, collate_fn = collate_graphs)
    tst_loader = GraphDataLoader(dataset = tst_dataset, batch_size = min([128, len(tst_dataset)]), shuffle = False, collate_fn = collate_graphs)

    task_type = dataset_1.task_type
    num_tasks = dataset_1.num_tasks

    node_dim = dataset_1.node_attr.shape[1]
    edge_dim = dataset_1.edge_attr.shape[1]

    gnn = GINPredictor(node_dim, edge_dim, n_tasks=num_tasks).cuda()

    model = Cons_Reg_Trainer(gnn, task_type, num_tasks)
    model.train(trn_loader, val_loader, model_path, args.cr_strength, args.num_views, args.flag_s)

    tst_y_pred = model.inference(tst_loader)
    tst_y = np.vstack([inst[-1] for inst in iter(tst_dataset)])

    if num_tasks > 1:
        
        test_results = []
        for i in range(num_tasks):
            tst_mask =np.isnan(tst_y[:,i])
            if len(np.unique(tst_y[:,i][~tst_mask]))==1:
                continue
            
            test_results.append(roc_auc_score(tst_y[:,i][~tst_mask], tst_y_pred[:,i][~tst_mask]))

        print('# of evaluated tasks:(%d/%d)'%(len(test_results), num_tasks))
        test_result = np.mean(test_results)

    else:
        tst_mask =np.isnan(tst_y).ravel()
        test_result = roc_auc_score(tst_y[~tst_mask], tst_y_pred[~tst_mask])

    print('--- test result:', test_result)

    if not os.path.exists('./result'):
        os.mkdir('./result')

    
    if not os.path.isfile('./result/result_%d_%d_%d_%d.csv'%(args.seed, args.split, args.num_views, args.flag_s)):
        with open('./result/result_%d_%d_%d_%d.csv'%(args.seed, args.split, args.num_views, args.flag_s), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'cvid', 'd_name', 'test_result'])

    with open('./result/result_%d_%d_%d_%d.csv'%(args.seed, args.split, args.num_views, args.flag_s), 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([args.seed, args.cvid, args.d_name, test_result])





if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--seed', type = int)
    arg_parser.add_argument('--d_path', type=str, default='./data/molnet_file/')
    arg_parser.add_argument('--d_name', type = str)
    arg_parser.add_argument('--backbone', type = str)
    arg_parser.add_argument('--cvid', type = int)
    arg_parser.add_argument('--split', type = int)

    arg_parser.add_argument('--cr_strength', type = float, default=1.)
    arg_parser.add_argument('--num_views', type = int, default=3)
    arg_parser.add_argument('--flag_s', type = int, default=2)

    args = arg_parser.parse_args()

    main(args)