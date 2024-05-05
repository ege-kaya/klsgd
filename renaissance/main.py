import torch
import torchvision
import torchvision.transforms as tvt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import pickle as pkl
import argparse
from opacus.grad_sample import GradSampleModule
from opacus.validators.module_validator import ModuleValidator
from networks import Linear, LeNet, PreActResNet18
from optimizers import KLSGD, WeightCalculator
from utils import MultiClassHingeLoss, train, make_noisy_data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import os

argparser = argparse.ArgumentParser()

'''
maxloss_hard: 4     minloss_hard: 5
maxloss_soft: 6     maxnorm_soft: 7
minloss_soft: 8     minnorm_soft: 9
maxnorm_hard: 10    minnorm_hard: 11
maxcorr_hard: 12    mincorr_hard: 13
maxcorr_topk: 14    mincorr_topk: 15
maxnorm_topk: 16    minnorm_topk: 17
maxloss_topk: 18    minloss_topk: 19
'''

argparser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adam", "poscorr_soft",
                                                              "maxcorr_soft", "mincorr_soft", 
                                                              "maxloss_hard", "minloss_hard",
                                                              "maxloss_soft", "maxnorm_soft",
                                                              "minloss_soft", "minnorm_soft",
                                                              "maxnorm_hard", "minnorm_hard",
                                                              "maxcorr_hard", "mincorr_hard",
                                                              "maxcorr_topk", "mincorr_topk",
                                                              "maxnorm_topk", "minnorm_topk",
                                                              "maxloss_topk", "minloss_topk"],
                                                              help="optimizer to use")
argparser.add_argument("--reg", default=1e-3, type=float, help="regularizer for KL term")
argparser.add_argument("--lr", default=1e-3, type=float, help="learning rate") # 1e-3
argparser.add_argument("--lr_decay", default=0.2, type=float, help="learning rate decay") # 0.2 
argparser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay parameter")
argparser.add_argument("--momentum", default=0.9, type=float, help="momentum parameter")
argparser.add_argument("--decay_schedule", default=30, type=int, help="decay schedule") # 10? 
argparser.add_argument("--epochs", default=50, type=int, help="number of epochs for training")
argparser.add_argument("--device_idx", default=0, type=int, help="cuda device idx")
argparser.add_argument("--batch_size", default=64, type=int, help="mini-batch size for training") # 64
argparser.add_argument("--num_workers", default=4, type=int, help="number of workers on cpu for dataloaders")
# argparser.add_argument("--seed", default=44, type=int, help="random seed")
argparser.add_argument("--dataset", default="MNIST", type=str, choices=["semeion", "MNIST",
                                                                        "KMNIST", "FMNIST",
                                                                        "CIFAR10", "CIFAR100",
                                                                        "SVHN"],
                                                                        help="dataset to train on")
argparser.add_argument("--model", default="LeNet", type=str, choices=["LeNet", "logistic", 
                                                                      "SVM", "ResNet18"])
argparser.add_argument("--aug", default=False, action='store_true', help="whether to augment data")
argparser.add_argument("--topk_ratio", default=1, type=float, help="ratio of the top-k elements chosen from the batch")

# file managment options 
argparser.add_argument("--save_path", default="./results", type=str, help="directory for the loss/accuracy history")

# noisy data settings 
argparser.add_argument("--noisy_data", action='store_true', help="whether to generate noisy data")
argparser.add_argument("--noise_frac", default=0.1, type=float, help="fraction of the noisy data")
argparser.add_argument("--noise_type", default="random", type=str, choices=("shift", "random"), help="type of the noise (i.e. shift or random)")


args = argparser.parse_args()

device = f'cuda:{args.device_idx}' if torch.cuda.is_available() else 'cpu'

optimizer = args.optimizer
lr = args.lr
reg = args.reg
topk_ratio = args.topk_ratio
data_path = "./data"
train_transform = tvt.Compose([tvt.ToTensor()])
test_transform = tvt.Compose([tvt.ToTensor()])
train_Sampler = None
test_Sampler = None
Shuffle = True
epochs = args.epochs
save_path = args.save_path #"./results"
#plot_path = "./plots"
seeds = [44, 45, 46] #47, 48]

os.makedirs(data_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
#os.makedirs(plot_path, exist_ok=True)


'''
DATASET SELECTION
'''

if args.dataset == 'MNIST':
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 50
    if args.aug:        
        end_epoch = 200 
        train_transform = tvt.Compose([
                            tvt.RandomCrop(28, padding=2),
                            tvt.RandomAffine(15, scale=(0.85, 1.15)),
                            tvt.ToTensor()       
                       ])                
    train_data = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'CIFAR10':
    nh = 32
    nw = 32
    nc = 3
    num_class = 10 
    end_epoch = 50
    if args.aug:
        end_epoch = 200 
        train_transform = tvt.Compose([
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor(),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'CIFAR100':
    nh = 32
    nw = 32
    nc = 3
    num_class = 100
    end_epoch = 50
    if args.aug:
        end_epoch = 200    
        train_transform = tvt.Compose([
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor()
        ])
        test_transform = tvt.Compose([
            tvt.ToTensor()
        ])
    train_data = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'SVHN':
    nh = 32
    nw = 32
    nc = 3
    num_class = 10
    end_epoch = 50    
    if args.aug:
        end_epoch = 200
        train_transform = tvt.Compose([
            tvt.RandomCrop(32, padding=4),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor()
        ])
        test_transform = tvt.Compose([
            tvt.ToTensor()
        ])
    train_data = torchvision.datasets.SVHN(data_path, split='train', download=True, transform=train_transform)
    test_data = torchvision.datasets.SVHN(data_path, split='test', download=True, transform=test_transform)    
elif args.dataset == 'FMNIST':
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 20
    if args.aug:
        end_epoch = 200       
        train_transform = tvt.Compose([
            tvt.RandomCrop(28, padding=2),
            tvt.RandomHorizontalFlip(),
            tvt.ToTensor()
        ]) 
    train_data = torchvision.datasets.FashionMNIST(data_path, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.FashionMNIST(data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'KMNIST':
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 50
    if args.aug:
        end_epoch = 200
        train_transform = tvt.Compose([
                            tvt.RandomCrop(28, padding=2),
                            tvt.ToTensor()       
                       ])
    train_data = torchvision.datasets.KMNIST(data_path, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.KMNIST(data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'semeion':
    nh = 16
    nw = 16
    nc = 1
    num_class = 10 # the digits from 0 to 9 (written by 80 people twice)    
    end_epoch = 50
    if args.aug:
        end_epoch = 200
        train_transform = tvt.Compose([
            tvt.RandomCrop(16, padding=1),
            tvt.RandomAffine(4, scale=(1.05, 1.05)),
            tvt.ToTensor()
        ])
        test_transform = tvt.Compose([
            tvt.ToTensor()
        ])    
    train_data = torchvision.datasets.SEMEION(data_path, transform=train_transform, download=True) 
    test_data = train_data    
    random_index = np.load(data_path+'/random_index.npy')
    train_size = 1000    
    train_Sampler = SubsetRandomSampler(random_index[range(train_size)])
    test_Sampler = SubsetRandomSampler(random_index[range(train_size,len(test_data))])
    Shuffle = False
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

model = args.model


'''
MODEL SELECTION
'''

def get_model(model_name):
    model = None
    if model_name == 'LeNet':
        model = LeNet(nc, nh, nw, num_class).to(device)
    elif model_name == 'ResNet18':
        model = PreActResNet18(nc, num_class).to(device)
    elif model_name == 'logistic' or args.model == 'SVM':
        dx = nh * nw * nc     
        model = Linear(dx, num_class).to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    return ModuleValidator.fix(model)

'''
LOSS SELECTION
'''

if args.model == "SVM":
    loss = MultiClassHingeLoss(reduction="none")
else:
    loss = nn.CrossEntropyLoss(reduction='none')



if __name__ == "__main__":
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmarks=False
        torch.autograd.set_detect_anomaly(True)

        
        # generate noisy data if wanted
        train_data = make_noisy_data(train_data, args.noise_type, args.noise_frac) if args.noisy_data else train_data

        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_Sampler, shuffle=Shuffle)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_Sampler, shuffle=False)

        model = get_model(args.model)
        if not args.optimizer == "sgd" and not args.optimizer == "adam":
            model = GradSampleModule(model)
                
        #optimizer = get_optimizer(args.optimizer, model)
        weight_calculator = WeightCalculator(params=model.parameters(), alg_name=args.optimizer, topk_ratio=args.topk_ratio, reg=args.reg)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=args.weight_decay, momentum=args.momentum)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_sch
        losses, accs = train(model, optimizer, weight_calculator, loss, train_loader, test_loader, epochs, args.optimizer, device)

        with open(f"{save_path}/acc_{args.optimizer}_{seed}_{lr}.pkl", "wb") as f:
            pkl.dump(accs, f)
        
        with open(f"{save_path}/loss_{args.optimizer}_{seed}_{lr}.pkl", "wb") as f:
            pkl.dump(losses, f)




