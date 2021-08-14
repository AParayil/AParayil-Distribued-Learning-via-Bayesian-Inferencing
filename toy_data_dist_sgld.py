import torch
from data.ToyDataset import ToyDataset
from models.toy_model import toy_model
from optimizers.dist_SGLD_toy import dsgld
from utils.getlaplacian import getlaplacian
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def adjust_lr(optimizer, modelparams, iterno, alpha0, beta0, gamma):
    alpha = alpha0 / ((1 + gamma * iterno) ** 0.55)
    beta = beta0 / ((1 + gamma * iterno) ** 0.04)
    for param_group in optimizer.param_groups:
        param_group['alpha'] = alpha
        param_group['beta'] = beta
        param_group['allmodels'] = modelparams

num_samples = 100
sigmax = math.sqrt(2.0)
sigma1 = math.sqrt(10.0)
sigma2 = math.sqrt(1.0)
toy_dataset = ToyDataset(num_samples=num_samples, theta1=0.0, theta2=1.0, sigmax=sigmax)

batch_size =  20
num_epochs = 1000000  
num_agents = 5
num_datasamples = int(len(toy_dataset) / num_agents)
lengths = [num_datasamples] * num_agents
num_mini_batches = math.ceil(num_datasamples / batch_size)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


train_loader = torch.utils.data.random_split(dataset=toy_dataset, lengths=lengths)

train_loader_list = []
for n in range(num_agents):
    train_loader_list.append(torch.utils.data.DataLoader(dataset=train_loader[n], batch_size=batch_size, shuffle=True))


model = []
modelparams = []
for k in range(num_agents):
    net = toy_model(sigmax=sigmax, sigma1=sigma1, sigma2=sigma2).to(device)
    model.append(net)
    modelparams.append(copy.deepcopy(net))

iterno = 1
gamma = 1 / 230.0
alpha0 = 0.2 / (230 ** 0.55)
adj, lap = getlaplacian(num_agents, type=0)
_, sv, _ = np.linalg.svd(lap)
beta0 = 1.01 / np.max(sv)
optimizerlist = []
for k in range(num_agents):
    optimizer = dsgld(model[k].parameters(), allmodels=modelparams, adj_vec=adj[k, :], alpha=alpha0, beta=beta0 , norm_sigma1=sigma1,
                      norm_sigma2 = sigma2, num_batches = len(train_loader_list[k]), addnoise=True)
    adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
    optimizerlist.append(optimizer)

PP = []
for epoch in range(1, num_epochs + 1):

    train_dataiter_list = []
    for k in range(num_agents):
        train_dataiter_list.append(iter(train_loader_list[k]))

    for i in range(num_mini_batches):

        iterno += 1

        modelparams = []
        for j in range(num_agents):

            cur_model = model[j]
            data = train_dataiter_list[j].next()
            data = data.to(device)
            optimizer = optimizerlist[j]
            optimizer.zero_grad()

            output = cur_model(data, device)
            loss = -output.sum()
            loss.backward()
            optimizer.step()

            modelparams.append(copy.deepcopy(model[j]))


        for k in range(num_agents):
            optimizer = optimizerlist[k]
            adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
            optimizerlist[k] = optimizer


    pp_epoch = []
    for k in range(num_agents):
        pp = []
        for p in model[k].parameters():
            pp.append(p.item())

        pp_epoch.append(np.asarray(pp))

    print('Epoch: {}. Parameters: {}'.format(epoch, np.asarray(pp_epoch)))

    PP.append(np.asarray(pp_epoch))
    print(iterno)

# Estimated Posterior
PP = np.asarray(PP)
fontsize = 10
lb = 0.25*num_epochs   #1000 # 000
ind = num_epochs*num_mini_batches-1
f2 = plt.figure()
ax = f2.gca()
pnum = 1  #151
for k in range(num_agents):
    PPplot = PP[lb:num_epochs:30, k, :] # 300
    axes = plt.subplot(2, 5, pnum)
    plt.scatter(PPplot[:, 0], PPplot[:, 1], s=0.1, c='k')
    plt.xlim(-1.5, 2.5)
    plt.ylim(-3, 3)
    pnum += 1


    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
