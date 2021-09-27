import torch
from utils.coco import COCO

from model.hourglass import get_hourglass
from model.MDN_hourglass import get_mixture_hourglass

from core.baseline_loss import _neg_loss,_reg_loss
from core.mixture_loss import mace_loss

from utils.utils import _tranpose_and_gather_feature
from core.utils import print_n_txt
import os

class SOLVER():
    def __init__(self,args):
        self.train_mode=True
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
        self.device = 'cuda'
        self.lr = 1e-3
        self.args=args
        self.load_dataset()
        self.load_model()

    def load_dataset(self):
        if self.train_mode:
            dataset = COCO(root='./cocodataset')
            self.dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0)
    
    def load_model(self):
        if self.args.base == 'baseline':
            self.model = get_hourglass['large_hourglass'].to(self.device)
            self.train=self.train_baseline
        elif self.args.base == 'mdn':
            self.model = get_mixture_hourglass['large_hourglass'].to(self.device)
            self.train = self.train_mdn
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
                    strtemp = ("iter: %d, loss: %.3f"%(e,loss))
                    print_n_txt(_f=f,_chars=strtemp)
                    strtemp = ("iter: %d, EPis : %.3f Alea: %.3f Pi entropy: %.3f"%(e,hmap_mace_loss['epis_avg']
                                                            ,hmap_mace_loss['alea_avg'],hmap_mace_loss['pi_entropy_avg']))
                    print_n_txt(_f=f,_chars=strtemp)
            total_loss /= len(self.dataloader)
            strtemp = ("EPOCH: %d LOSS: %.3f"%(epoch,total_loss))
            print_n_txt(_f=f,_chars=strtemp)
            torch.save(self.model.state_dict(),'./ckpt/baseline/{}.pt'.format(epoch))
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