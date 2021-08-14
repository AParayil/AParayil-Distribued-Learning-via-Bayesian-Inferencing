import torch
import torch.nn
from data.a9aDataset import a9aDataset
from models.a9a_model import a9a_model
from optimizers.dist_SGLD_a9a import dsgld
from utils.getlaplacian import getlaplacian
import torch.utils.data
import numpy as np
import math
import ipdb
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import load_svmlight_file


def adjust_lr(optimizer, modelparams, iterno, alpha0, beta0, gamma):
    alpha = alpha0 / ((1 + gamma * iterno) ** 0.55)
    beta = beta0 / ((1 + gamma * iterno) ** 0.05)
    for param_group in optimizer.param_groups:
        param_group['alpha'] = alpha
        param_group['beta'] = beta
        param_group['allmodels'] = modelparams


def accuracy_evaluation(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    for data, labels in data_loader:
        data = data.to(device)

        outputs = net(data)

        predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).float()

        if torch.cuda.is_available():
            correct += predicted.cpu().eq(labels.view_as(predicted.cpu())).sum().item()
        else:
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

        total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# Read data
Xtr, Ytr = load_svmlight_file('./data/a9a')
Xtr = Xtr.todense()
indx = Ytr==-1
Ytr[indx] = 0
total_num_samples = Xtr.shape[0]
num_feat = Xtr.shape[1]
train_num_samples = 26050
test_num_samples = 6511
# torch.manual_seed(1)
batch_size = 10
num_epochs = 2
num_agents = 5
run_size = 50
test_accuracy_agr = np.zeros((105, num_agents, run_size))
mean_accuracy = np.zeros((105, num_agents))
std_accuracy = np.zeros((105, num_agents))
num_datasamples = int(train_num_samples / num_agents)
lengths = [num_datasamples] * num_agents
num_mini_batches = math.ceil(num_datasamples / batch_size)
gamma = 1 / 230.0
alpha0 = 0.004 / ((230 ** 0.55) * num_agents)
#ipdb.set_trace()
adj, lap = getlaplacian(num_agents, type=0)
_, sv, _ = np.linalg.svd(lap)
beta0 = 1.01 / np.max(sv)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

for run in range(0, run_size):
    # Randomly sample training and test data
    total_idx = np.random.choice(total_num_samples, total_num_samples, replace=False)
    train_idx = total_idx[:train_num_samples]
    test_idx = total_idx[-test_num_samples:]
    # Train and test data sets
    train_data = a9aDataset(data=Xtr, labels=Ytr, train_idx=train_idx, test_idx=test_idx, train=True)
    test_data = a9aDataset(data=Xtr, labels=Ytr, train_idx=train_idx, test_idx=test_idx, train=False)

     # Train and test data loaders
    train_data_split = torch.utils.data.random_split(dataset=train_data, lengths=lengths)

    train_loader_list = []
    for k in range(num_agents):
        train_loader_list.append(torch.utils.data.DataLoader(dataset=train_data_split[k], batch_size=batch_size,
                                                         shuffle=True))

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Model and loss
    model = []
    modelparams = []
    for k in range(num_agents):
        net = a9a_model(num_inputs=num_feat).to(device)
        model.append(net)
        modelparams.append(copy.deepcopy(net))

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    # Optimizer
    optimizerlist = []
    iterno = 1
    for k in range(num_agents):
        optimizer = dsgld(model[k].parameters(), allmodels=modelparams, adj_vec=adj[k, :], alpha=alpha0, beta=beta0 ,
                      weight_decay=True, num_batches = len(train_loader_list[k]), addnoise=True)
        adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
        optimizerlist.append(optimizer)

    # Test accuracy before training
    net_test_accuracy = []
    for k in range(num_agents):
        net_test_accuracy.append(accuracy_evaluation(model[k], test_dataloader, device))
    test_accuracy = [net_test_accuracy]
    iterlist = [iterno]

# Training
    for epoch in range(1, num_epochs + 1):

        train_dataiter_list = []
        for k in range(num_agents):
            train_dataiter_list.append(iter(train_loader_list[k]))

        for i in range(num_mini_batches):

            iterno += 1
            modelparams = []
            for j in range(num_agents):

                cur_model = model[j]
                cur_model.train()
                data, labels = train_dataiter_list[j].next()
                data = data.to(device)
                labels = labels.to(device)
                optimizer = optimizerlist[j]
                optimizer.zero_grad()
                output = cur_model(data)
                loss = criterion(output.squeeze(), labels)
                loss.backward()
                optimizer.step()
                modelparams.append(copy.deepcopy(model[j]))

            for k in range(num_agents):
                optimizer = optimizerlist[k]
                adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
                optimizerlist[k] = optimizer

            if iterno % 10 == 0:
                net_test_accuracy = []
                for k in range(num_agents):
                    net_test_accuracy.append(accuracy_evaluation(model[k], test_dataloader, device))
                test_accuracy.append(net_test_accuracy)
                print('Iteration: {}. Accuracy: {}'.format(iterno, np.asarray(net_test_accuracy)))
                iterlist.append(iterno)
    test_accuracy_agr[:, :, run] = test_accuracy

# Results
mean_accuracy = np.mean(test_accuracy_agr, axis=2)
std_accuracy = np.std(test_accuracy_agr, axis=2)
f1 = plt.figure()
ax = f1.gca()
plt.plot(np.asarray(iterlist), np.asarray(mean_accuracy))
plt.grid()
fontsize = 10
ax.fill_between(np.asarray(iterlist), (np.asarray(mean_accuracy[:, 1])-np.asarray(std_accuracy)[:, 1]), (np.asarray(mean_accuracy[:, 1])+np.asarray(std_accuracy[:, 1])), color='b', alpha=.5)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim(57, 87)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
plt.show()

