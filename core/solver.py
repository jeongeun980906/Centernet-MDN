import torch
from utils.coco import COCO,COCO_eval

from model.hourglass import get_hourglass
from model.MDN_hourglass import get_mixture_hourglass

from core.baseline_loss import _neg_loss,_reg_loss
from core.mixture_loss import mace_loss,_gather

from utils.utils import _tranpose_and_gather_feature
from utils.image import transform_preds
from utils.post_process import ctdet_decode
import numpy as np
from core.utils import print_n_txt
import os

class SOLVER():
    def __init__(self,args):
        self.train_mode=args.train
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
        self.device = 'cuda'
        self.lr = 1e-3
        self.args=args
        self.load_dataset()
        self.load_model()
        if not self.train:
            self.load_ckpt(0)

    def load_dataset(self):
        if self.train_mode:
            self.dataset = COCO(root='./cocodataset')
            self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=8,shuffle=True,num_workers=0)
        else:
            self.dataset = COCO_eval(root='./cocodataset',split='test')
            self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=1,shuffle=True,num_workers=0)

    def load_model(self):
        if self.args.base == 'baseline':
            self.model = get_hourglass['large_hourglass'].to(self.device)
            self.train=self.train_baseline
            self.test = self.test_baseline
        elif self.args.base == 'mdn':
            self.model = get_mixture_hourglass['large_hourglass'].to(self.device)
            self.train = self.train_mdn
            self.test = self.test_mdn
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr,weight_decay=1e-7)
    
    def train_mdn(self):
        print("TRAIN")
        EPOCHS = 20
        txtName = ('./res/mdn_log.txt')
        f = open(txtName,'w') # Open txt file
        for epoch in range(EPOCHS):
            total_loss=0
            for e,batch in enumerate(self.dataloader):
                outputs = self.model(batch['image'].to(self.device))
                hmaps, regs, w_h_ = zip(*outputs)
                regs = [_tranpose_and_gather_feature(r, batch['inds'].to(self.device)) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds'].to(self.device)) for r in w_h_]
                
                hmap_loss = 0
                for hmap in hmaps:
                    pi,mu,sigma = hmap['pi'], hmap['mu'], hmap['sigma']
                    hmap_mace_loss = mace_loss(pi,mu,sigma, batch['hmap'].to(self.device))
                    hmap_loss += hmap_mace_loss['mace_avg'] - 1 * hmap_mace_loss['epis_avg']  + 1* hmap_mace_loss['alea_avg']
                reg_loss = _reg_loss(regs, batch['regs'].to(self.device), batch['ind_masks'].to(self.device))
                w_h_loss = _reg_loss(w_h_, batch['w_h_'].to(self.device), batch['ind_masks'].to(self.device))
                loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss
                if e%50==0:
                    strtemp = ("iter: %d, loss: %.3f hmap loss: %.3f reg_loss: %.3f w_h_loss: %.3f "%(e,loss,hmap_loss,reg_loss,w_h_loss))
                    print_n_txt(_f=f,_chars=strtemp)
                    strtemp = ("iter: %d, EPis : %.3f Alea: %.3f Pi entropy: %.3f"%(e,hmap_mace_loss['epis_avg']
                                                            ,hmap_mace_loss['alea_avg'],hmap_mace_loss['pi_entropy_avg']))
                    print_n_txt(_f=f,_chars=strtemp)
            total_loss /= len(self.dataloader)
            strtemp = ("EPOCH: %d LOSS: %.3f"%(epoch,total_loss))
            print_n_txt(_f=f,_chars=strtemp)
            torch.save(self.model.state_dict(),'./ckpt/mdn/{}.pt'.format(epoch))
    def train_baseline(self):
        print("TRAIN")
        EPOCHS = 20
        txtName = ('./res/baseline_log.txt')
        f = open(txtName,'w') # Open txt file
        for epoch in range(EPOCHS):
            total_loss=0
            for e,batch in enumerate(self.dataloader):
                outputs = self.model(batch['image'].to(self.device))
                hmap, regs, w_h_ = zip(*outputs)
                regs = [_tranpose_and_gather_feature(r, batch['inds'].to(self.device)) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds'].to(self.device)) for r in w_h_]

                hmap_loss = _neg_loss(hmap, batch['hmap'].to(self.device))
                reg_loss = _reg_loss(regs, batch['regs'].to(self.device), batch['ind_masks'].to(self.device))
                w_h_loss = _reg_loss(w_h_, batch['w_h_'].to(self.device), batch['ind_masks'].to(self.device))
                loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss
                if e%50==0:
                    strtemp = ("iter: %d, loss: %.3f"%(e,loss))
                    print_n_txt(_f=f,_chars=strtemp)
            total_loss /= len(self.dataloader)
            strtemp = ("EPOCH: %d LOSS: %.3f"%(epoch,total_loss))
            print_n_txt(_f=f,_chars=strtemp)
            torch.save(self.model.state_dict(),'./ckpt/baseline/{}.pt'.format(epoch))
    
    def test_baseline(self):
        max_per_image = 100
        txtName = ('./res/baseline_test.txt')
        f = open(txtName,'w') # Open txt file
        self.model.eval()
        results = {}
        with torch.no_grad():
            for e,(img_id, inputs) in enumerate(self.dataloader):
                if e%50==0:
                    print("test on Image ID: {} / {}".format(e,len(self.dataloader)))
                detections = []
                for scale in inputs:
                    inputs[scale]['image'] = inputs[scale]['image'].to(self.device)
                    output = self.model(inputs[scale]['image'].squeeze(1))[-1]
                    dets = ctdet_decode(*output)
                    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                    top_preds = {}
                    # print(inputs[scale]['scale'])
                    dets[:, :2] = transform_preds(dets[:, 0:2],
                                                inputs[scale]['center'],
                                                inputs[scale]['scale'],
                                                (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                                inputs[scale]['center'],
                                                inputs[scale]['scale'],
                                                (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    cls = dets[:, -1]
                    for j in range(self.dataset.num_classes):
                        inds = (cls == j)
                        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                        top_preds[j + 1][:, :4] /= scale

                    detections.append(top_preds)

                bbox_and_scores = {}
                for j in range(1, self.dataset.num_classes + 1):
                    bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
                scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, self.dataset.num_classes + 1)])

                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, self.dataset.num_classes + 1):
                        keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                        bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

                results[img_id] = bbox_and_scores

            eval_results = self.dataset.run_eval(results)
            strtemp = (eval_results)
            print_n_txt(_f=f,_chars=strtemp)
            
    def load_ckpt(self,epoch):
        path = './ckpt/{}/{}.pt'.format(self.args.base,epoch)
        state_dict=torch.load(path)
        self.model.load_state_dict(state_dict)

    def test_mdn(self):
        txtName = ('./res/mdn_test.txt')
        f = open(txtName,'w') # Open txt file
        max_per_image = 100
        self.model.eval()
        results = {}
        with torch.no_grad():
            for e,(img_id, inputs) in enumerate(self.dataloader):
                if e%50==0:
                    print("test on Image ID: {} / {}".format(e,len(self.dataloader)))
                detections = []
                for scale in inputs:
                    inputs[scale]['image'] = inputs[scale]['image'].to(self.device)
                    output = self.model(inputs[scale]['image'].squeeze(1))[-1]
                    pi = output[0]['pi']
                    mu = output[0]['mu']
                    sigma = output[0]['sigma']
                    output[0] = _gather(pi,mu,sigma)['mu_prime']
                    dets = ctdet_decode(*output)
                    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                    top_preds = {}
                    # print(inputs[scale]['scale'])
                    dets[:, :2] = transform_preds(dets[:, 0:2],
                                                inputs[scale]['center'],
                                                inputs[scale]['scale'],
                                                (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                                inputs[scale]['center'],
                                                inputs[scale]['scale'],
                                                (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    cls = dets[:, -1]
                    for j in range(self.dataset.num_classes):
                        inds = (cls == j)
                        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                        top_preds[j + 1][:, :4] /= scale

                    detections.append(top_preds)

                bbox_and_scores = {}
                for j in range(1, self.dataset.num_classes + 1):
                    bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
                scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, self.dataset.num_classes + 1)])

                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, self.dataset.num_classes + 1):
                        keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                        bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

                results[img_id] = bbox_and_scores

            eval_results = self.dataset.run_eval(results)
            strtemp = (eval_results)
            print_n_txt(_f=f,_chars=strtemp)