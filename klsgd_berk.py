
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch import nn
import numpy as np
import torch 
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import random 
import argparse 
from pathlib import Path 
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.validators.module_validator import ModuleValidator
import matplotlib.pyplot as plt
import pickle as pkl


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    # Just return the number of files in the root folder
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])


class MNIST_ResNet(nn.Module):
    def __init__(self):
        super(MNIST_ResNet, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        return self.model(x)
    

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5) # 28
        self.conv2 = nn.Conv2d(6, 16, 5) # 10
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
 
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 14
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # 5
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

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
                """
                # only sample with the largest gradient norm 
                elif self.alg_no == 10:
                    norms = torch.norm(torch.flatten(p.grad_sample, 1), p=2, dim=1)
                    idx = torch.argmax(norms)
                    mask = torch.zeros_like(norms)
                    mask[idx] = 1
                    grad_weights = mask
                # only sample with the smallest gradient norm 
                elif self.alg_no == 11:
                    norms = torch.norm(torch.flatten(p.grad_sample, 1), p=2, dim=1)
                    idx = torch.argmin(norms)
                    mask = torch.zeros_like(norms)
                    mask[idx] = 1
                    grad_weights = mask
                # only sample with the largest positive correlation 
                elif self.alg_no == 12:
                    dot_product = torch.sum(torch.flatten(p.grad) * torch.flatten(p.grad_sample, 1), dim=1)
                    idx = torch.argmax(dot_product)
                    mask = torch.zeros_like(dot_product)
                    mask[idx] = 1
                    grad_weights = mask
                # only sample with the largest negative correlation
                elif self.alg_no == 13:
                    dot_product = torch.sum(torch.flatten(-p.grad) * torch.flatten(p.grad_sample, 1), dim=1)
                    idx = torch.argmax(dot_product)
                    mask = torch.zeros_like(dot_product)
                    mask[idx] = 1
                    grad_weights = mask
                """

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

def train_and_evaluate(model, optimizer, scheduler, loss_fn, save_dir, train_dataloader, val_dataloader, epochs, step_size, early, alg_no):

    train_loss = []
    test_acc = []
    # Training routine
    # Set some initially high previous loss values, for choosing of minimum
    # These parameters involve the early stopping mechanism
    prev_class_loss = 1e5
    violations = 0
    
    # Keep track of the classification and regression losses for plotting later.
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    
    # Summary writer for saving losses and heatmaps 
    writer = SummaryWriter(save_dir)
    
    for epoch in range(epochs):
        model.train()
        scheduler.step()
        # Keeping track of running losses.
        class_running_loss = 0.
        for i, data in enumerate(train_dataloader):
            
            # Get the inputs and the labels, put them on cuda
            inputs, label_gts = data
            inputs = inputs.to(DEVICE)
            label_gts = label_gts.to(DEVICE)
            
            # Get the outputs from the model
            label_pred = model(inputs)
            # Do the backpropagation with respect to the classification loss, but do not break down the
            # computational graph!
            if args.optimizer == "klsgd" and (alg_no in (4,5,6,8)):
                class_loss = F.cross_entropy(label_pred, label_gts, reduction="none")
                optimizer.sample_losses = class_loss
                class_loss = class_loss.mean()
            else:
                class_loss = F.cross_entropy(label_pred, label_gts)
                optimizer.zero_grad()

            class_loss.backward()
            optimizer.step()
            
            # Update the running losses.
            #print("Mean Loss:", torch.mean(class_loss).item())
            class_running_loss += torch.mean(class_loss).item()
    
            # Just some user feedback reporting the losses.
            if (i+1) % 50 == 0:
                mean_loss = class_running_loss / 50
                if (i+1) % 100 == 0:
                    print("[epoch: %d, batch: %5d] class. loss: %.3f" % (epoch+1, i+1, mean_loss))
                writer.add_scalar('training loss', mean_loss, epoch * len(train_dataloader) + i)
                class_running_loss = 0.0
        
        # Validation for early stopping. We will stop training if the validation losses do not decrease for
        # a specified number of consecutive epochs. This will save us time on the training and prevent overfitting
        # to the training data.
        train_loss.append(mean_loss)
        with torch.no_grad():
            model.eval()
            total_class_loss = 0
            correct = 0
            total = 0
            cf_total = 0
            # Exactly the same stuff we do in training, just on the validation data and with no training, only
            # evaluation!
            for i, data in enumerate(test_dataloader):
                inputs, label_gts = data
                inputs = inputs.to(DEVICE)
                label_gts = label_gts.to(DEVICE)
                label_pred = model(inputs)
                total_class_loss += torch.mean(loss_fn(label_pred, label_gts)).item()
                
                # Check the Accuracy of the model 
                _, label_pred = torch.max(label_pred, dim=1)
                # Update the tallies for total examples considered and correctly classified
                correct += torch.eq(label_gts, label_pred).sum()
                total += label_gts.shape[0]
                # Calculate confusion matrix, we can compute each batch's confusion matrix separately
                # and accumulate them over the batches.
                cf = confusion_matrix(label_gts.cpu().detach().numpy(), label_pred.cpu().detach().numpy(), labels=np.arange(2))
                cf_total += cf
            
            # At the end, calculate validation accuracy
            accuracy = correct / total
            test_acc.append(accuracy.item())
            print("test accuracy", accuracy)
            print("test loss", total_class_loss / len(test_dataloader))

            # # Checking if our validation losses have actually increased after the training.
            # if total_class_loss < prev_class_loss:
            
            #     # Found a new best model, so save it!
            #     print("Found new best model")
            #     torch.save(model.state_dict(), save_dir / "model_best")
            #     prev_class_loss = total_class_loss
                
            #     # Reset the early stopping counter.
            #     violations = 0
                
            #     # Epoch of the current best model found.
            #     stopped = epoch + 1
                
            # # If the model hasn't improved, increase the early stopping counter.
            # else:
            #     violations += 1
                
            # # If the early stopping criteria is met, abandon training!
            # if violations == early:
            #     print("Training stopped at", epoch+1)
            #     break
    return train_loss, test_acc, cf_total


if __name__ == "__main__":
    NAME = "sgd"
    parser = argparse.ArgumentParser()
    # KLSGD parameters 
    parser.add_argument("--optimizer", default="sgd", type=str, choices=("sgd", "adam", "klsgd"), help="type of the optimizer")
    parser.add_argument("--robust", default=False, action='store_true', help="switch for being robust or performance oriented")
    parser.add_argument("--alg_no", default=-1, type=int, choices=(4,5,6,7,8,9,10,11,12,13,14,15,16,17-1), help="algorithm number")
    parser.add_argument("--topk", default=None, type=float, help="ratio of the top-k elements chosen from the batch")
    parser.add_argument("--reg", default=1e-3, type=float, help="regularizer for KL term")
    parser.add_argument("--adam", default=False, action='store_true', help="whether to use Adam or not for KLSGD")
    # classic training parameters
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default=0.1, type=float, help="learning rate decay")
    parser.add_argument("--decay_schedule", default=10, type=int, help="decay schedule")
    parser.add_argument("--epochs", default=25, type=int, help="number of epochs for training")
    parser.add_argument("--device_idx", default=0, type=int, help="cuda device idx")
    parser.add_argument("--batch_size", default=32, type=int, help="mini-batch size for training")
    parser.add_argument("--early", default=5, type=int, help="early stopping limit")

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers on cpu for dataloaders")
    # file managment 
    parser.add_argument("--save_dir", default="exps/new_exp", type=str, help="save directory for the experiment")
    parser.add_argument("--mixing", default=3000, type=int, help="how many correctly labeled samples from each class")
    args = parser.parse_args()
    seed = 101
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)
   
    DEVICE = f"cuda:{args.device_idx}"
    convert = tvt.Compose([tvt.ToTensor(), tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # resnet = MNIST_ResNet().to(DEVICE)

    resnet = LeNet()
    resnet = ModuleValidator.fix(resnet)

    if args.optimizer == "klsgd":
        NAME = "klsgd_robust" if args.robust else "klsgd_performance"
        resnet = GradSampleModule(resnet)
    
    if args.alg_no != -1:
        NAME = f"alg{args.alg_no}"

    data = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=convert, download=True)

    preprocess_dataloader = DataLoader(data, batch_size=1)

    train_data = []
    train_labels = []

    val_data = []
    val_labels = []

    test_data = []
    test_labels = []

    one_count = 0
    seven_count = 0
    for datum, target in preprocess_dataloader:
        if target == 1:
            if one_count < args.mixing:
                train_data.append(datum.squeeze())
                train_labels.append(0)
                one_count += 1
            elif one_count < args.mixing + 500:
                val_data.append(datum.squeeze())
                val_labels.append(0)
                one_count += 1
            elif one_count < args.mixing + 1000:
                test_data.append(datum.squeeze())
                test_labels.append(0)
                one_count += 1
            else:
                train_data.append(datum.squeeze())
                train_labels.append(1)

        elif target == 7:
            if seven_count < args.mixing:
                train_data.append(datum.squeeze())
                train_labels.append(1)
                seven_count += 1
            elif seven_count < args.mixing + 500:
                val_data.append(datum.squeeze())
                val_labels.append(1)
                seven_count += 1
            elif seven_count < args.mixing + 1000:
                test_data.append(datum.squeeze())
                test_labels.append(1)
                seven_count += 1
            else:
                train_data.append(datum.squeeze())
                train_labels.append(0)
        
        if len(train_data) == 8000 and len(val_data) == 1000 and len(test_data) == 1000:
            break
    
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    # train_dataset = Dataset(train_data)
    # val_dataset = Dataset(val_data)
    # test_dataset = Dataset(test_data)

    train_dataset = MyDataset(train_data, train_labels)
    val_dataset = MyDataset(val_data, val_labels)
    test_dataset = MyDataset(test_data, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    # print(args.robust)

    if args.optimizer == "klsgd":
        print("Using KLSGD")
        optimizer = KLSGD(resnet.parameters(), robust=args.robust, lr=args.lr, reg=args.reg, alg_no=args.alg_no, topk=args.topk)
    elif args.optimizer == "sgd":
        print("Using SGD")
        optimizer = torch.optim.SGD(resnet.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    # do file managment 
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_schedule, gamma=args.lr_decay)

    train_loss, test_acc, cf = train_and_evaluate(
        model=resnet, 
        optimizer=optimizer, 
        scheduler=scheduler,
        loss_fn=loss_fn,
        save_dir=save_dir,
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        epochs=args.epochs, 
        step_size=args.lr, 
        early=args.early, 
        alg_no=args.alg_no
    )

    with open(save_dir / f"loss_{NAME}.pkl", "wb") as f:
        pkl.dump(train_loss, f)
    
    s = sns.heatmap(cf, annot=True, fmt='d')
    fig = s.get_figure()
    fig.savefig(save_dir / f"cf_{NAME}.png", dpi=400, bbox_inches="tight")
    plt.close()

    with open(save_dir / f"acc_{NAME}.pkl", "wb") as f:
        pkl.dump(test_acc, f)
