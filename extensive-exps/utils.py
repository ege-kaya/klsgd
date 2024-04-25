import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


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
    
def train(model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs, alg, device):

    train_loss = []
    test_acc = []

    # Keep track of the classification and regression losses for plotting later.
    model = model.to(device)
    print(f"Using device: {device}")
        
    for epoch in range(epochs):
        model.train()
        # Keeping track of running losses.
        class_running_loss = 0.
        for i, data in enumerate(train_dataloader):
            
            # Get the inputs and the labels, put them on cuda
            inputs, label_gts = data
            inputs = inputs.to(device)
            label_gts = label_gts.to(device)
            
            # Get the outputs from the model
            label_pred = model(inputs)

            if alg in ["maxloss_hard", "minloss_hard", "maxloss_soft", "minloss_soft", "maxloss_topk", "minloss_topk"]:
                class_loss = loss_fn(label_pred, label_gts)
                optimizer.sample_losses = class_loss
                class_loss = class_loss.mean()
            else:
                class_loss = loss_fn(label_pred, label_gts).mean()
            
            
            optimizer.zero_grad()
            class_loss.backward()
            optimizer.step()
            # scheduler.step()
            
            # Update the running losses.
            class_running_loss += torch.mean(class_loss).item()
    
            # Just some user feedback reporting the losses.
            if (i+1) % 50 == 0:
                mean_loss = class_running_loss / 50
                if (i+1) % 100 == 0:
                    print("[epoch: %d, batch: %5d] class. loss: %.3f" % (epoch+1, i+1, mean_loss))
                class_running_loss = 0.0
                train_loss.append(mean_loss)
        
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

    return train_loss, test_acc

def plot(save_path, plot_path, seeds, alg):
    loss = 0
    loss_arr = []
    for seed in seeds:
        full_name = f"{save_path}/loss_{alg}_{seed}.pkl"

        with open(full_name, "rb") as f:
            loss = np.array(pkl.load(f))
            loss_arr.append(loss)

    std = np.std(loss_arr, axis=0)
    mean = np.mean(loss_arr, axis=0)
    lower = mean - std
    upper = mean + std

    plt.fill_between(upper, lower, alpha=.2, linewidth=0)
    plt.plot(mean)
    plt.title("Training loss vs epochs")
    plt.grid("major")
    plt.savefig(f"{plot_path}/loss_{alg}.png", dpi=400, bbox_inches="tight")
    plt.close()

    acc = 0
    acc_arr = []
    for seed in seeds:
        full_name = f"{save_path}/acc_{alg}_{seed}.pkl"

        with open(full_name, "rb") as f:
            acc = np.array(pkl.load(f))
            acc_arr.append(acc)


    std = np.std(acc_arr, axis=0)
    mean = np.mean(acc_arr, axis=0)
    lower = mean - std
    upper = mean + std

    plt.fill_between(upper, lower, alpha=.2, linewidth=0)
    plt.plot(mean)
    plt.title("Test accuracy vs epochs")
    plt.grid("major")
    plt.savefig(f"{plot_path}/acc_{alg}.png", dpi=400, bbox_inches="tight")


