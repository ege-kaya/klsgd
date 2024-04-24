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
from optimizers import KLSGD
from utils import MultiClassHingeLoss, train, plot
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
'''

argparser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adam", 
                                                              "klsgd_r", "klsgd_p", 
                                                              "maxloss_hard", "minloss_hard",
                                                              "maxloss_soft", "maxnorm_soft",
                                                              "minloss_soft", "minnorm_soft",
                                                              "maxnorm_hard", "minnorm_hard",
                                                              "maxcorr_hard", "mincorr_hard",
                                                              "maxcorr_topk", "mincorr_topk",
                                                              "maxnorm_topk", "minnorm_topk"],
                                                              help="optimizer to use")
argparser.add_argument("--reg", default=1e-5, type=float, help="regularizer for KL term")
argparser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
argparser.add_argument("--lr_decay", default=0.1, type=float, help="learning rate decay")
argparser.add_argument("--decay_schedule", default=10, type=int, help="decay schedule")
argparser.add_argument("--epochs", default=25, type=int, help="number of epochs for training")
argparser.add_argument("--device_idx", default=0, type=int, help="cuda device idx")
argparser.add_argument("--batch_size", default=32, type=int, help="mini-batch size for training")
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
argparser.add_argument("--topk_ratio", default=0.25, type=float, help="ratio of the top-k elements chosen from the batch")


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
save_path = "./results"
plot_path = "./plots"
seeds = [44, 45, 46, 47, 48]

os.makedirs(data_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)


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

train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_Sampler, shuffle=Shuffle)
test_loader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_Sampler, shuffle=False)

model = args.model


'''
MODEL SELECTION
'''

if model == 'LeNet':
    model = LeNet(nc, nh, nw, num_class).to(device)
elif model == 'ResNet18':
    model = PreActResNet18(nc, num_class).to(device)
elif model == 'logistic' or args.model == 'SVM':
    dx = nh * nw * nc     
    model = Linear(dx, num_class).to(device)
else:
    raise ValueError(f"Invalid model: {args.model}")


model = ModuleValidator.fix(model)


'''
OPTIMIZER SELECTION
'''

if optimizer == "sgd":
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
elif optimizer == "adam":
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
elif optimizer == "klsgd_r":
    optimizer = KLSGD(params=model.parameters(), lr=lr, reg=reg, robust=True)
elif optimizer == "klsgd_p":
    optimizer = KLSGD(params=model.parameters(), lr=lr, reg=reg)
elif optimizer == "maxloss_hard":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=4)
elif optimizer == "minloss_hard":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=5)
elif optimizer == "maxloss_soft":
    optimizer = KLSGD(params=model.parameters(), lr=lr, reg=reg, alg_no=6)
elif optimizer == "maxnorm_soft":
    optimizer = KLSGD(params=model.parameters(), lr=lr, reg=reg, alg_no=7)
elif optimizer == "minloss_soft":
    optimizer = KLSGD(params=model.parameters(), lr=lr, reg=reg, alg_no=8)
elif optimizer == "minnorm_soft":
    optimizer = KLSGD(params=model.parameters(), lr=lr, reg=reg, alg_no=9)
elif optimizer == "maxnorm_hard":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=10)
elif optimizer == "minnorm_hard":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=11)
elif optimizer == "maxcorr_hard":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=12)
elif optimizer == "mincorr_hard":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=13)
elif optimizer == "maxcorr_topk":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=14, topk=topk_ratio)
elif optimizer == "mincorr_topk":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=15, topk=topk_ratio)
elif optimizer == "maxnorm_topk":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=16, topk=topk_ratio)
elif optimizer == "minnorm_topk":
    optimizer = KLSGD(params=model.parameters(), lr=lr, alg_no=17, topk=topk_ratio)
else:
    raise ValueError(f"Invalid optimizer: {args.optimizer}")

if not args.optimizer == "sgd" and not args.optimizer == "adam":
    model = GradSampleModule(model)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_schedule, gamma=args.lr_decay)


'''
LOSS SELECTION
'''

if args.model == "SVM":
    loss = MultiClassHingeLoss()
else:
    loss = nn.CrossEntropyLoss(reduction='mean')



if __name__ == "__main__":
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmarks=False
        torch.autograd.set_detect_anomaly(True)

        losses, accs = train(model, optimizer, scheduler, loss, train_loader, test_loader, epochs, args.optimizer, device)

        with open(f"{save_path}/acc_{args.optimizer}_{seed}.pkl", "wb") as f:
            pkl.dump(accs, f)
        
        with open(f"{save_path}/loss_{args.optimizer}_{seed}.pkl", "wb") as f:
            pkl.dump(losses, f)

    plot(save_path, plot_path, seeds, args.optimizer)





