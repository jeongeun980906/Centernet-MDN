import numpy as np
import torch
import torch.nn as nn

from model.hourglass import *

class MixtureHead(nn.Module):
    def __init__(self,
                 in_dim     = 64,   # input feature dimension 
                 cnv_dim = 256,     # cnv dim
                 num_classes      = 80,   # number of classes 
                 k          = 5,    # number of mixtures
                 sig_min    = 1, # minimum sigma
                 sig_max    = 10, # maximum sigma
                 SHARE_SIG  = True  # share sigma among mixture
                 ):
        super(MixtureHead,self).__init__()
        self.in_dim     = in_dim    # Q
        self.num_classes= num_classes     # D
        self.k          = k         # K
        self.sig_min    = sig_min
        self.sig_max    = sig_max
        self.SHARE_SIG  = SHARE_SIG
        self.cnv_dim    = cnv_dim
        self.build_graph()

    def build_graph(self):
        self.cnn_pi      = make_kp_layer(self.cnv_dim, self.in_dim, self.k)
        self.cnn_mu      = make_kp_layer(self.cnv_dim, self.in_dim, self.k*self.num_classes)
        if self.SHARE_SIG:
            self.cnn_sigma   = make_kp_layer(self.cnv_dim, self.in_dim, self.k)
        else:
            self.cnn_sigma   = make_kp_layer(self.cnv_dim, self.in_dim, self.k*self.num_classes)
        self.cnn_mu[-1].bias.data.fill_(-2.19)
        self.cnn_sigma[-1].bias.data.fill_(-2.19)

    def forward(self,x):
        """
            :param x: [N x Q]
        """
        pi_logit        = self.cnn_pi(x)                                 # [N x K x W x H]
        pi              = torch.softmax(pi_logit,dim=1)                 # [N x K x W x H]
        # print(pi.size())
        mu              = self.cnn_mu(x)                                 # [N x KD x W x H]
        mu              = torch.reshape(mu,(-1,self.k,self.num_classes,pi.size(2),pi.size(3)))      # [N x K x D x W x H]
        # print(mu.size())
        if self.SHARE_SIG:
            sigma       = self.cnn_sigma(x)                              # [N x K x W x H]
            sigma       = sigma.unsqueeze(dim=2)                       # [N x K x 1 x W x H]
            sigma       = sigma.expand_as(mu)                           # [N x K x D x W x H]
        else:
            sigma       = self.cnn_sigma(x)                              # [N x KD x W x H]
        sigma           = torch.reshape(sigma,(-1,self.k,self.num_classes,pi.size(2),pi.size(3)))   # [N x K x D x W x H]
        if self.sig_max is None:
            sigma = self.sig_min + torch.exp(sigma)                     # [N x K x D]
        else:
            sig_range = (self.sig_max-self.sig_min)
            sigma = self.sig_min + sig_range*torch.sigmoid(sigma)       # [N x K x D]
        # print(sigma.size())
        mol_out = {'pi':pi,'mu':mu,'sigma':sigma}
        return mol_out

class MDN_hourglass(nn.Module):
    def __init__(self,n,nstack,dims,modules,cnv_dim=256,num_classes=80):
        super(MDN_hourglass,self).__init__()
        self.nstack = nstack
        self.num_classes = num_classes
        curr_dim = dims[0]
        
        self.pre = nn.Sequential(convolution(7,3,128,stride=2),
                                residual(3,128,curr_dim,stride=2))
        self.kps = nn.ModuleList([kp_module(n,dims,modules) for _ in range(nstack)])
        self.cnvs = nn.ModuleList([convolution(3,curr_dim,cnv_dim) for _ in range(nstack)])
        self.inters = nn.ModuleList([residual(3,curr_dim,curr_dim) for _ in range(nstack-1)])

        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim,curr_dim,(1,1),bias=False),
                                        nn.BatchNorm2d(curr_dim)) for _ in range(nstack-1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim,curr_dim,(1,1),bias=False),
                                        nn.BatchNorm2d(curr_dim)) for _ in range(nstack-1)])
        # Heatmap Mixture
        self.hmap = nn.ModuleList([MixtureHead(in_dim=curr_dim,cnv_dim=cnv_dim,num_classes=num_classes) for _ in range(nstack)])

        # regression layers
        self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.w_h_ = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

        self.relu = nn.ReLU(inplace=True)
    def forward(self, image):
        inter = self.pre(image) 
        outs = []
        for ind in range(self.nstack):
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if self.training or ind == self.nstack - 1:
                outs.append([self.hmap[ind](cnv), self.regs[ind](cnv), self.w_h_[ind](cnv)])

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

get_mixture_hourglass = \
  {'large_hourglass':
     MDN_hourglass(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4],num_classes=80),
   'small_hourglass':
     MDN_hourglass(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4],num_classes=80)}

if __name__ == '__main__':

  net = get_mixture_hourglass['large_hourglass']
  net=net.to('cuda')
  with torch.no_grad():
    outputs = net(torch.randn(2, 3, 512, 384).to('cuda'))
  hmap, regs, w_h_= zip(*outputs)
  print("hmap[0]  : pi: {} mu: {}, sigma:{}".format(hmap[0]['pi'].shape,hmap[0]['mu'].shape,hmap[0]['sigma'].shape))
  print("hmap[1]  : pi: {} mu: {}, sigma:{}".format(hmap[1]['pi'].shape,hmap[1]['mu'].shape,hmap[1]['sigma'].shape))
  print("regs  : ",regs[0].shape, regs[1].shape)
  print("w_h_  : ",w_h_[0].shape, w_h_[1].shape)