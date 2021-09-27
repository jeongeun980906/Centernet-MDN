import numpy as np
import torch

device='cuda'
def np2tc(x_np): return torch.from_numpy(x_np).float().to(device)
def tc2np(x_tc): return x_tc.detach().cpu().numpy()

def _gather(pi,mu,sigma):
    """
        :param pi:      [N x K x W x H]
        :param mu:      [N x K x D x W x H]
        :param sigma:   [N x K x D x W x H]
    """
    max_idx = torch.argmax(pi,dim=1) # [N x W x H]
    mu      = torch.sigmoid(mu,dim=2) # [N x K x D x W x H]
    idx_gather = max_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2],mu.shape[3],mu.shape[4]).unsqueeze(1) # [N x 1 x D x W x H]
    mu_sel = torch.gather(mu,dim=1,index=idx_gather).squeeze(dim=1) # [N x D x W x H]
    sigma_sel = torch.gather(sigma,dim=1,index=idx_gather).squeeze(dim=1) # [N x D x W x H]
    out = {'max_idx':max_idx, # [N x W x H]
           'mu_sel':mu_sel, # [N x D x W x H]
           'sigma_sel':sigma_sel # [N x D x W x H]
           }
    return out

def mace_loss(pi,mu,sigma,target):
    """
        :param pi:      [N x K x W x H]
        :param mu:      [N x K x D x W x H]
        :param sigma:   [N x K x D x W x H]
        :param target:  [N x D x W x H]
    """
    # $\pi$
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1 x W x H]
    pi_exp = pi_usq.expand_as(mu) # [N x K x D x W x H]

    ne_loss = neg_loss(mu,target) # [N x K x D x W x H]
    ace_exp = ne_loss / sigma # attenuated CE [N x K x D x W x H]
    mace_exp = torch.mul(pi_exp,ace_exp) # mixtured attenuated CE [N x K x D x W x H]
    mace = torch.sum(mace_exp,dim=1) # [N x D x W x H]
    mace = torch.sum(torch.sum(torch.sum(mace,dim=1),dim=-1),dim=-1) # [N]
    mace_avg = torch.mean(mace) # [1]
    # Compute uncertainties (epis and alea)
    unct_out = mln_uncertainties(pi,mu,sigma)
    epis = unct_out['epis'] # [N x W x H]
    alea = unct_out['alea'] # [N x W x H]
    pi_entropy = unct_out['pi_entropy'] # [N x W x H]
    epis_avg = torch.mean(epis) # [1]
    alea_avg = torch.mean(alea) # [1]
    pi_entropy_avg=torch.mean(pi_entropy) # [1]
    # Return
    loss_out = {'mace':mace, # [N]
                'mace_avg':mace_avg, # [1]
                'epis':epis, # [N x W x H]
                'alea':alea, # [N x W x H]
                'epis_avg':epis_avg, # [1]
                'alea_avg':alea_avg, # [1],
                'pi_entropy':pi_entropy, # [N x W x H]
                'pi_entropy_avg':pi_entropy_avg # [1]
                }
    return loss_out

def mln_uncertainties(pi,mu,sigma):
    """
        :param pi:      [N x K x W x H]
        :param mu:      [N x K x D x W x H]
        :param sigma:   [N x K x D x W x H]
    """
    # entropy of pi
    entropy_pi  = -pi*torch.log(pi+1e-8)
    entropy_pi  = torch.sum(entropy_pi,1) # [N x W x H]
    # $\pi$
    mu_hat = torch.sigmoid(mu) # logit to prob [N x K x D x W x H]
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1 x W x H]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D x W x H]
    # sigmoid($\mu$) average
    mu_hat_avg = torch.sum(torch.mul(pi_exp,mu_hat),dim=1).unsqueeze(1) # [N x 1 x D x W x H]
    mu_hat_avg_exp = mu_hat_avg.expand_as(mu) # [N x K x D x W x H]
    mu_hat_diff_sq = torch.square(mu_hat-mu_hat_avg_exp) # [N x K x D x W x H]
    # Epistemic uncertainty
    epis = torch.sum(torch.mul(pi_exp,mu_hat_diff_sq), dim=1)  # [N x D x W x H]
    epis = torch.sqrt(torch.sum(epis,dim=1)+1e-6) # [N x W x H]
    # Aleatoric uncertainty
    alea = torch.sum(torch.mul(pi_exp,sigma), dim=1)  # [N x D x W x H]
    alea = torch.sqrt(torch.mean(alea,dim=1)+1e-6) # [N x W x H]
    # Return
    unct_out = {'epis':epis, # [N x W x H]
                'alea':alea,  # [N x W x H]
                'pi_entropy':entropy_pi
                }
    return unct_out

def mln_eval(pi,mu,sigma,num,N=10):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    top_pi,top_idx = torch.topk(pi,num,dim=1) # [N X n]
    top_pi=torch.softmax(top_pi,dim=-1)
    max_idx = torch.argmax(pi,dim=1) # [N]
    max2_idx= top_idx[:,1] # [N]

    mu      = torch.softmax(mu,dim=2) # [N x K x D]
    mu_max = torch.argmax(mu,dim=2) # [N x K]
    mu_onehot=_to_one_hot(mu_max,N)

    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]

    mu_exp = torch.mul(pi_exp,mu_onehot) # mixtured mu [N x K x D]
    mu_prime = torch.sum(mu_exp,dim=1) # [N x D]

    sig_exp = torch.mul(pi_exp,sigma) # mixtured mu [N x K x D]
    sig_prime = torch.sum(sig_exp,dim=1) # [N x D]

    idx1_gather = max_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel = torch.gather(mu,dim=1,index=idx1_gather).squeeze(dim=1) # [N x D]
    sigma_sel = torch.gather(sigma,dim=1,index=idx1_gather).squeeze(dim=1) # [N x D]

    idx2_gather = max2_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel2 = torch.gather(mu,dim=1,index=idx2_gather).squeeze(dim=1) # [N x D]
    sigma_sel2 = torch.gather(sigma,dim=1,index=idx2_gather).squeeze(dim=1) # [N x D]
   
    unct_out = mln_uncertainties(pi,mu,sigma)
    pi_entropy = unct_out['pi_entropy'] # [N]

    out = {'max_idx':max_idx, # [N]
           'mu_sel':mu_sel, # [N x D]
           'sigma_sel':sigma_sel, # [N x D]
           'mu_sel2':mu_sel2, # [N x D]
           'sigma_sel2':sigma_sel2, # [N x D]
           'mu_prime': mu_prime, # [N x D]
           'sigma_prime': sig_prime, # [N x D]
           'pi_entropy': pi_entropy, # [N]
           'top_pi':top_pi
           }
    return out


def neg_loss(preds, targets):
    '''
    preds: [N x K x D x W x H]
    targets: [N x D x W x H]
    loss: [N x K x D x W x H]
    '''

    target_usq = torch.unsqueeze(targets,1) # [N x 1 x D x W x H]
    targets = target_usq.expand_as(preds) # [N x K x D x W x H]
    
    pos_inds = targets == 1
    neg_inds = targets < 1 

    pred = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)
    pos_pred = torch.ones_like(pred)
    pos_pred[pos_inds] = pred[pos_inds] # [N x K x D x W x H]
    
    neg_pred = torch.zeros_like(pred)
    neg_pred[neg_inds] = pred[neg_inds] # [N x K x D x W x H]

    neg_weights = torch.zeros_like(pred)
    neg_weights[neg_inds] = torch.pow(1 - targets[neg_inds], 4)
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    if pos_pred.nelement() == 0:
        loss = - neg_loss
    else:
        loss = - (pos_loss + neg_loss) / num_pos
    return loss