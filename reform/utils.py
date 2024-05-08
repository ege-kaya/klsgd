import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from pathlib import Path 
import copy 
import random
import torch.nn.functional as F 


class MultiClassHingeLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(MultiClassHingeLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, y):
        index = torch.arange(0, y.size()[0]).long()
        output_y = output[index, y.data].view(-1,1)
        loss = output - output_y + 1.0 
        loss[index, y.data] = 0
        loss[loss < 0] = 0
        if self.reduction == "mean":
            loss = torch.sum(loss, dim=1) / output.size()[1]

        return loss 

def output(data_loader, model, alg, device, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0    
    total_correct = 0      
    total_size = 0   
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        h1 = model(x)
        y_hat = h1.max(1)[1]
        if alg == 'SVM':
            total_loss += torch.mean(hinge_loss(h1, y)).item() * y.size(0)
        else:                
            total_loss += F.cross_entropy(h1, y).item() * y.size(0)
        total_correct += y_hat.eq(y.data).cpu().sum()                
        total_size += y.size(0)    
    # print
    total_loss /= total_size 
    total_acc = 100. * float(total_correct) / float(total_size)  
    return (total_loss, total_acc)  
    
def train(model, optimizer, weight_calculator, loss_fn, train_dataloader, test_dataloader, epochs, alg, device, adaptive):

    train_loss, train_acc, test_loss, test_acc = [],[0.],[],[0.]
    init_topk_ratio = weight_calculator.topk_ratio
    # Keep track of the classification and regression losses for plotting later.
    model = model.to(device)
    print(f"Using device: {device}")

    for epoch in range(epochs):
        if adaptive:
            if train_acc[-1] >= 99.5:
                weight_calculator.topk_ratio = init_topk_ratio / 16
            elif train_acc[-1] >= 95.:
                weight_calculator.topk_ratio = init_topk_ratio / 8
            elif train_acc[-1] >= 90.:
                weight_calculator.topk_ratio = init_topk_ratio / 4
            elif train_acc[-1] >= 80.:
                weight_calculator.topk_ratio = init_topk_ratio / 2

        model.train()
        # Keeping track of running losses.
        class_running_loss = 0.
        for i, data in enumerate(train_dataloader):
            
            # Get the inputs and the labels, put them on cuda
            inputs, label_gts = data
            inputs = inputs.to(device)
            label_gts = label_gts.to(device)
            
            # Get the outputs from the model
            model.enable_hooks()
            model.zero_grad()
            label_pred = model(inputs)
            sample_loss = loss_fn(label_pred, label_gts)
            mean_loss = torch.mean(sample_loss)
            mean_loss.backward(retain_graph=True)

            weight_calculator.sample_losses = sample_loss.detach()
            weights = weight_calculator.calc_weights()
            model.disable_hooks()
            model.zero_grad()
            # label_pred = model(inputs)
            # sample_loss2 = loss_fn(label_pred, label_gts)
            new_loss = torch.sum(weights.to(device) * sample_loss)
            new_loss.backward()

            optimizer.step()
            # scheduler.step()
            
            # Update the running losses.
            class_running_loss += torch.mean(sample_loss).item()
    
            # Just some user feedback reporting the losses.
            if (i+1) % 50 == 0:
                mean_loss = class_running_loss / 50
                if (i+1) % 100 == 0:
                    print("[epoch: %d, batch: %5d] class. loss: %.3f" % (epoch+1, i+1, mean_loss))
                class_running_loss = 0.0
                #train_loss.append(mean_loss)
        optimizer = lr_scheduler(optimizer, epoch + 1)
        # evaluate the loss & accuracy of the model over train & test datasets 
        (tr_loss, tr_acc) = output(train_dataloader, model, alg, device, train=True)
        (te_loss, te_acc) = output(test_dataloader, model, alg, device, train=False)
        # save the losses & accuracies 
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(te_loss)
        test_acc.append(te_acc)

        # display the values 
        print("*"*20)
        print(f"train accuracy: {tr_acc}")
        print(f"test accuracy: {te_acc}")
        print("*"*20)

    return train_loss, test_acc


def plot_loss(fig, save_path, plot_path, seeds, alg):
    loss = 0
    loss_arr = []
    for seed in seeds:
        full_name = f"{save_path}/loss_{alg}_{seed}_0.001.pkl"

        with open(full_name, "rb") as f:
            loss = np.array(pkl.load(f))
            loss_arr.append(loss)

    std = np.std(loss_arr, axis=0)
    mean = np.mean(loss_arr, axis=0)
    lower = mean - std
    upper = mean + std

    plt.fill_between(upper, lower, alpha=.2, linewidth=0)
    fig.plot(mean)
    return fig 
    #plt.title("Training loss vs epochs")
    #plt.grid("major")
    #plt.savefig(f"{plot_path}/loss_{alg}.png", dpi=400, bbox_inches="tight")
    #plt.close()


def plot_acc(fig, save_path, plot_path, seeds, alg):
    acc = 0
    acc_arr = []
    for seed in seeds:
        full_name = f"{save_path}/acc_{alg}_{seed}_0.001.pkl"

        with open(full_name, "rb") as f:
            acc = np.array(pkl.load(f))
            acc_arr.append(acc)

    std = np.std(acc_arr, axis=0)
    mean = np.mean(acc_arr, axis=0)
    lower = mean - std
    upper = mean + std

    fig.fill_between(upper, lower, alpha=.2, linewidth=0)
    fig.plot(mean)
    #plt.title("Test accuracy vs epochs")
    #plt.grid("major")
    #plt.savefig(f"{plot_path}/acc_{alg}.png", dpi=400, bbox_inches="tight")
    return fig 

def make_noisy_data(train_data, noise_type, noise_frac):
    '''
    This function takes a Dataset and make it noisy Dataset
    by adding noise to the features.
    '''
    # get dataset parameters 
    data_size = len(train_data)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=data_size, shuffle=True)
    (data, target) = next(iter(dataloader))

    # make features noisy 
    flag = np.random.binomial(1, noise_frac, size=(data_size, 1))
    for idx, val in enumerate(flag):
        datapoint = data[idx]
        if val[0] == 1 and noise_type == "feature_add":
            noise = 100 * torch.randn_like(datapoint)
            data[idx] += noise
        elif val[0] == 1 and noise_type == "feature_imp":
            mask_flags = torch.rand_like(data[idx])
            mask = torch.zeros_like(data[idx])
            mask[mask_flags <= 0.9] = 1.
            mask[mask_flags <= 0.45] = -1.
            data[idx] += mask
            data[idx] = torch.clamp(data[idx], min=0., max=1.)
        else:
            continue 
    print("Data shape:", data.shape)
    noisy_data = torch.utils.data.TensorDataset(data, target)
    return noisy_data

if __name__ == '__main__':

    save_path = Path("./results")
    plot_path = Path("./plots")
    seeds = [44,45,46] #47, 48

    plt.figure()
    plt.title("Training loss vs epochs (FMNIST, LeNet)")
    plt.grid("major")
    plt.xlabel("epochs")
    plt.ylabel("cross-entropy loss")
    alg_list = ["maxnorm_topk", "maxnorm_soft", "maxnorm_hard", "maxloss_topk",
                "maxloss_soft", "maxloss_hard", "maxcorr_topk", "maxcorr_hard",
                "klsgd_r", "sgd"]
    for alg in alg_list:
        loss = 0
        loss_arr = []
        for seed in seeds:
            full_name = f"{save_path}/loss_{alg}_{seed}_0.001.pkl"

            with open(full_name, "rb") as f:
                loss = np.array(pkl.load(f))
                loss_arr.append(loss)

        std = np.std(loss_arr, axis=0)
        mean = np.mean(loss_arr, axis=0)
        lower = mean - std
        upper = mean + std

        #plt.fill_between(upper, lower, alpha=.2, linewidth=0)
        plt.plot(mean)

    plt.legend(alg_list)
    plt.savefig(f"{plot_path}/loss.png", dpi=400, bbox_inches="tight")
    plt.close()


    # ACCURACY
    plt.figure()
    plt.title("Test accuracy vs epochs (FMNIST, LeNet)")
    plt.grid("major")
    plt.xlabel("epochs")
    plt.ylabel("classification accuracy %")
    alg_list = ["maxnorm_topk", "maxnorm_soft", "maxnorm_hard", "maxloss_topk",
                "maxloss_soft", "maxloss_hard", "maxcorr_topk", "maxcorr_hard",
                "klsgd_r", "sgd"]
    for alg in alg_list:
        loss = 0
        loss_arr = []
        for seed in seeds:
            full_name = f"{save_path}/acc_{alg}_{seed}_0.001.pkl"

            with open(full_name, "rb") as f:
                loss = np.array(pkl.load(f))
                loss_arr.append(loss)

        std = np.std(loss_arr, axis=0)
        mean = np.mean(loss_arr, axis=0)
        lower = mean - std
        upper = mean + std

        #plt.fill_between(upper, lower, alpha=.2, linewidth=0)
        plt.plot(mean)

    plt.legend(alg_list)
    plt.savefig(f"{plot_path}/acc.png", dpi=400, bbox_inches="tight")
    plt.close()

def lr_decay_func(optimizer, lr_decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer    
def lr_scheduler(optimizer, epoch, lr_decay=0.1, interval=10):
    #if args.data_aug == 0:
    if epoch == 10 or epoch == 50:
        optimizer = lr_decay_func(optimizer, lr_decay=lr_decay) 
    #if args.data_aug == 1:
    #    if epoch == 10 or epoch == 100:
    #        optimizer = lr_decay_func(optimizer, lr_decay=lr_decay)                   
    return optimizer


'''
        with torch.no_grad():
            model.eval()
            total_class_loss = 0
            correct = 0
            total = 0
            # Exactly the same stuff we do in training, just on the validation data and with no training, only
            # evaluation!
            for i, data in enumerate(test_dataloader):
                inputs, label_gts = data
                inputs = inputs.to(device)
                label_gts = label_gts.to(device)
                label_pred = model(inputs)
                total_class_loss += torch.mean(loss_fn(label_pred, label_gts)).item()
                
                # Check the Accuracy of the model 
                _, label_pred = torch.max(label_pred, dim=1)
                # Update the tallies for total examples considered and correctly classified
                correct += torch.eq(label_gts, label_pred).sum()
                total += label_gts.shape[0]
            
            # At the end, calculate validation accuracy
            accuracy = correct / total
            test_acc.append(accuracy.item())
            print("test accuracy", accuracy)
            print("test loss", total_class_loss / len(test_dataloader))
        '''