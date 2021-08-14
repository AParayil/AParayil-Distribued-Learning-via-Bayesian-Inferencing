import torch
from data.ToyDataset import ToyDataset
from models.toy_model import toy_model
from optimizers.SGLD_toy import sgld
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.ticker import MaxNLocator


def adjust_lr(optimizer, iter, a, b, gamma):
    lr = a / ((b + iter) ** gamma)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


num_samples = 100
batch_size = 1   
#torch.manual_seed(1)
sigmax = math.sqrt(2.0)
sigma1 = math.sqrt(10.0)
sigma2 = math.sqrt(1.0)
torch.manual_seed(0)
num_epochs = 1000000 
a = 0.2
b = 230
gamma = 0.55
iterno = 1
itertn=[]
stoch_grad = np.zeros((1, 2))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

toy_dataset = ToyDataset(num_samples=num_samples, theta1=0.0, theta2=1.0, sigmax=sigmax)
toy_dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=batch_size, shuffle=True)

model = toy_model(sigmax=sigmax, sigma1=sigma1, sigma2=sigma2).to(device)
optimizer = sgld(model.parameters(), lr=0.01, norm_sigma1=sigma1, norm_sigma2=sigma2, num_batches = num_samples/batch_size, addnoise=True)
adjust_lr(optimizer, iter=iterno, a=a, b=b, gamma=gamma)

PP = []
for epoch in range(1, num_epochs + 1):
    model.train()
    tot_loss = 0.0
    for data in toy_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, device)
        loss = -output.sum() # Negative log likelihood
        loss.backward()
        optimizer.step()
        ##
        itertn.append(iterno)
        tot_loss += loss.item()
        iterno += 1
        ##
        adjust_lr(optimizer, iter=iterno, a=a, b=b, gamma=gamma)
        pp = []
        for p in model.parameters():
            pp.append(p.item())
        PP.append(np.asarray(pp))

    print(tot_loss / (num_samples/batch_size), "   ", pp[0], "   ", pp[1])

fontsize = 10
PP = np.asarray(PP)
lb = math.ceil(num_epochs*0.1)
PPplot = PP[100000:1000000:250, :]
#lb:num_epochs:300, :]
f1 = plt.figure()
plt.scatter(PPplot[:, 0], PPplot[:, 1], s=0.1, c='k')
plt.xlim(-1.5, 2.5)
plt.ylim(-3, 3)
ax = f1.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()



# True posterior
x = toy_dataset.data.numpy().reshape(-1)

th1, th2 = np.mgrid[-1:2:.01, -2:2:.01]
th12 = th1 + th2
pos = np.zeros_like(th1)

for i in range(th1.shape[0]):
    for j in range(th1.shape[1]):
        lh = np.prod(0.5 * multivariate_normal.pdf(x, mean=th1[i, j], cov=2) + \
             0.5 * multivariate_normal.pdf(x, mean=th12[i, j], cov=2))
        pos[i, j] = multivariate_normal.pdf(th1[i, j], mean=0.0, cov=10) * \
                    multivariate_normal.pdf(th2[i, j], mean=0.0, cov=1) * lh

f3 = plt.figure()
ax = f3.gca()
plt.contour(th1, th2, pos)
plt.xlim(-1.5, 2.5)
plt.ylim(-3, 3)
fontsize = 10
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()









