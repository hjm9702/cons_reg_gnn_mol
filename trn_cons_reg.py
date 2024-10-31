import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from scipy.special import expit



class Cons_Reg_Trainer():

    def __init__(self, gnn, task_type, num_tasks):
        self.gnn = gnn
        self.task_type = task_type
        self.num_tasks = num_tasks
        
        self.cuda = torch.device('cuda:0')



    def train(self, trn_loader, val_loader, model_path, cr_strength, num_views, flag_s):
        max_epochs = 300
        init_lr = 5e-4
        patience = 50
        
        optimizer = Adam(self.gnn.parameters(), lr = init_lr, weight_decay = 1e-5)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        val_log = np.zeros(max_epochs)

        print('cr_strength:', cr_strength)
        print('num_views:', num_views)
        print('flag s: ', flag_s)

        for epoch in range(max_epochs):

            self.gnn.train()
            start_time = time.time()
            
            for batchidx, batchdata in enumerate(trn_loader):

                inputs, inputs_aug, ys = batchdata
                mask = torch.isnan(ys)
                
                inputs = inputs.to(self.cuda)
                inputs_aug = [input_aug.to(self.cuda) for input_aug in inputs_aug]
                ys = ys.to(self.cuda)
                ys_imputed = torch.nan_to_num(ys,0).to(self.cuda)
                mask = mask.to(self.cuda)
                    
                
                if self.num_tasks == 1:
                    ys = ys.unsqueeze(1)
                    ys_imputed = ys_imputed.unsqueeze(1)
                    mask = mask.unsqueeze(1)

                
                optimizer.zero_grad()

                step_size = 5e-3
                perturb = torch.FloatTensor(inputs.ndata['node_attr'].shape).uniform_(-step_size, step_size).cuda()
                perturb.requires_grad_()

                pred_graph_feats, pred = self.gnn(inputs, perturb)

                cls_loss = loss_fn(pred, ys_imputed)
                cls_loss = torch.where(~mask, cls_loss, torch.nan)
                cls_loss = cls_loss.nanmean(dim=1).nanmean()

                cls_loss /= flag_s

                for _ in range(flag_s-1):
                    cls_loss.backward()
                    perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                    perturb.data = perturb_data.data
                    perturb.grad[:] = 0

                    pred_graph_feats, pred = self.gnn(inputs, perturb)

                    cls_loss = loss_fn(pred, ys_imputed)
                    cls_loss = torch.where(~mask, cls_loss, torch.nan)
                    cls_loss = cls_loss.nanmean(dim=1).nanmean()
                    
                    cls_loss /= flag_s

                res_augs = [self.gnn(input_aug) for input_aug in inputs_aug]

                pred_graph_feats_augs = [self.gnn.project(res_aug[0]) for res_aug in res_augs]

                pred_det = pred.detach()
                pred_det = torch.sigmoid(pred_det)

                pred_prob = (ys_imputed.long()*pred_det + (1-ys_imputed.long())*(1-pred_det))
                pred_prob = torch.where(~mask, pred_prob, torch.nan).nanmean(1)
                
                pred_graph_feats_det = self.gnn.project(pred_graph_feats).detach()
                
                reg_loss = torch.mean(torch.stack([(1-torch.nn.CosineSimilarity(dim=1)(val,pred_graph_feats_det)) for val in pred_graph_feats_augs],1),1)

                reg_loss = reg_loss[pred_prob>0.95].mean()
                if torch.isnan(reg_loss): reg_loss=0

                loss = cls_loss+cr_strength*reg_loss
                    
                loss.backward()
                optimizer.step()

                train_loss = loss.detach().item()
                
            # validation
            if self.num_tasks > 1:
                val_y = np.vstack([inst[-1] for inst in iter(val_loader.dataset)])
            else:
                val_y = np.hstack([inst[-1] for inst in iter(val_loader.dataset)])
            
            val_y = torch.Tensor(val_y)
            val_y_imputed = torch.nan_to_num(val_y,0)
            val_mask = torch.isnan(val_y)

            val_y_pred = self.inference(val_loader)

            val_loss = loss_fn(torch.Tensor(val_y_pred), val_y_imputed)
            val_loss = torch.where(~val_mask, val_loss, torch.nan)
            val_loss = val_loss.nanmean(dim=-1).mean().detach().item()


            val_log[epoch] = val_loss
            if epoch%10 == 0:
                print('--- validation epoch %d, processed %d, train_cls_loss %.3f, train_cr_loss %.3f, train_loss %.3f, current loss %.3f, best loss %.3f time elapsed(min) %.2f'%(epoch, val_loader.dataset.__len__(), cls_loss, cr_strength*reg_loss, train_loss, val_loss, np.min(val_log[:epoch+1]), (time.time()-start_time)/60))

            if np.argmin(val_log[:epoch+1]) == epoch:
                torch.save(self.gnn.state_dict(), model_path)

                self.epoch_num = epoch

            elif np.argmin(val_log[:epoch+1]) <= epoch - patience:
                break
            
        print('best epoch: %d!'%self.epoch_num)
        
        self.gnn.load_state_dict(torch.load(model_path))


    
    def inference(self, loader):

        self.gnn.eval()
        tst_y_pred = []

        with torch.no_grad():
            for batchidx, batchdata in enumerate(loader):
                inputs = batchdata[0].to(self.cuda)

                _, pred = self.gnn(inputs)

                tst_y_pred += pred.cpu()

        
        if self.num_tasks > 1: tst_y_pred = np.vstack(tst_y_pred)
        else: tst_y_pred = np.hstack(tst_y_pred)

        tst_y_pred = expit(tst_y_pred)

        return tst_y_pred





