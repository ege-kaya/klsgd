import torch
import torch.nn.functional as F

class Adam(object):
    def __init__(self, betas, weight_decay=0):
        self.b1, self.b2 = betas 
        self.weight_decay = weight_decay
        self.m = []
        self.v = []
        self.t = 0

    def get_update(self, update, params, idx):
        # initialize moment terms with 0 tensors if not done 
        if self.t == 1:
            self.m.append(torch.zeros_like(update))
            self.v.append(torch.zeros_like(update))
            idx = -1
        # get the recent moment terms for the given layer
        m = self.m[idx]
        v = self.v[idx]

        if self.weight_decay != 0:
            update.add_(self.weight_decay * params)
        # first moment update 
        m = self.b1 * m + (1 - self.b1) * update 
        # second moment update 
        v = self.b2 * v + (1 - self.b2) * torch.pow(update, 2)
        m_unbias = m / (1 - self.b1 ** self.t)
        v_unbias = v / (1 - self.b2 ** self.t)
        new_update = torch.div(m_unbias, torch.sqrt(v_unbias) + 1e-6)
        # update the moments 
        self.m[idx] = m
        self.v[idx] = v
        return new_update

        

class KLSGD(torch.optim.Optimizer):
    def __init__(self, params, robust=True, lr=1e-3, reg=0.5, adam=False, alg_no=-1, topk=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if reg < 0.0:
            raise ValueError(f"Invalid regularization weight: {reg}")

        # Dictionary to store gradients for each parameter
        defaults = dict(lr=lr, reg=reg)
        self.lr = lr
        self.reg = reg
        self.robust = robust
        self.alg_no = alg_no
        self.topk_ratio = topk
        super(KLSGD, self).__init__(params, defaults)
        self.dir = -1 if self.robust else 1
        self.adam = Adam(betas=(.9, .99)) if adam else None 
        if alg_no in (4,5,6,8):
            self.sample_losses = None

    def calc_weights(self):
        weights = []

        if self.alg_no in (10,11,12,13,14,15):
            total_weight = 0 

        layer_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                if type(p.grad_sample) is list:
                    p.grad_sample = p.grad_sample[-1]

                layer_count += 1
                # only sample with the largest/smallest gradient norm 
                if self.alg_no in (10, 11, 16, 17):
                    total_weight += torch.sum(torch.flatten(p.grad_sample, 1) ** 2, dim=1)
                # only sample with the largest/negative positive correlation 
                elif self.alg_no in (12, 13, 14, 15):
                    total_weight += torch.sum(torch.flatten(p.grad) * torch.flatten(p.grad_sample, 1), dim=1)
                # only sample with the worst loss used 
                if self.alg_no == 4:
                    idx = torch.argmax(self.sample_losses)
                    mask = torch.zeros_like(self.sample_losses)
                    mask[idx] = 1
                    grad_weights = mask
                # only sample with the best loss used
                elif self.alg_no == 5:
                    idx = torch.argmin(self.sample_losses)
                    mask = torch.zeros_like(self.sample_losses)
                    mask[idx] = 1
                    grad_weights = mask
                # using loss in exponential 
                elif self.alg_no == 6:
                    energy = self.sample_losses / self.reg
                    grad_weights = F.softmax(energy, dim=0)
                # using gradient norm in exponential
                elif self.alg_no == 7:
                    energy = torch.norm(torch.flatten(p.grad_sample, 1), p=2, dim=1)**2 / self.reg
                    grad_weights = F.softmax(energy, dim=0)
                # using negative loss in exponential
                elif self.alg_no == 8:
                    energy = -self.sample_losses / self.reg
                    grad_weights = F.softmax(energy, dim=0)
                # using negative gradient norm in exponential
                elif self.alg_no == 9:
                    energy = -torch.norm(torch.flatten(p.grad_sample, 1), p=2, dim=1)**2 / self.reg
                    grad_weights = F.softmax(energy, dim=0)
                else:
                    # KLSGD algorithm 
                    dot_product = torch.sum(torch.flatten(p.grad) * torch.flatten(p.grad_sample, 1), dim=1)
                    energy = self.dir * self.lr * dot_product / self.reg # torch.exp(self.dir * self.lr * dot_product / self.reg)
                    # print(energy)
                    grad_weights = F.softmax(energy, dim=0)

                    
                if self.alg_no not in (10,11,12,13):
                    weights.append(grad_weights)

        if self.alg_no in (10, 12):
            idx = torch.argmax(total_weight)
            mask = torch.zeros_like(total_weight)
            mask[idx] = 1 
            weights = torch.tile(mask[None,:], (layer_count, 1))
        # choose maximum top-k elements
        elif self.alg_no in (14,16):
            _, indices = torch.topk(total_weight, k=int(self.topk_ratio * len(total_weight)))
            mask = torch.zeros_like(total_weight)
            mask[indices] = 1 
            weights = torch.tile(mask[None,:], (layer_count, 1))
        elif self.alg_no in (11, 13):
            idx = torch.argmin(total_weight)
            mask = torch.zeros_like(total_weight)
            mask[idx] = 1 
            weights = torch.tile(mask[None,:], (layer_count, 1))
        # choose minimum top-k elements
        elif self.alg_no in (15, 17):
            _, indices = torch.topk(total_weight, k=int(self.topk_ratio * len(total_weight)), largest=False)
            mask = torch.zeros_like(total_weight)
            mask[indices] = 1 
            weights = torch.tile(mask[None,:], (layer_count, 1))

        return weights


    def step(self, closure=None):
        if self.adam is not None:
            self.adam.t += 1
        # calculate the mean gradient for mini-batch 
        weights = self.calc_weights()
        idx = -1
        for group in self.param_groups:
            for p in group["params"]:
                idx += 1
                weight = weights[idx]
                update = torch.zeros_like(p.grad)
                
                if type(p.grad_sample) is list:
                    p.grad_sample = p.grad_sample[-1]

                product = weight.view(-1, *([1] * (p.grad_sample.dim() - 1))) * p.grad_sample
                update.add_(torch.sum(product, axis=0))

                if self.adam is None:
                    p.data.add_(-self.lr * update)
                else:
                    p.data.add_(-self.lr * self.adam.get_update(update, p.data, idx))
        return weights