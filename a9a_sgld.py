import torch
import torch.nn
from data.a9aDataset import a9aDataset
from models.a9a_model import a9a_model
from optimizers.SGLD_a9a import sgld
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file


def adjust_lr(optimizer, iterno, a, b, gamma):
    lr = a / ((b + iterno) ** gamma)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_evaluation(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    for data, labels in data_loader:
        data = data.to(device)

        outputs = net(data)

        predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).float()

        correct += predicted.cpu().eq(labels.view_as(predicted.cpu())).sum().item()

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
run_size = 50
num_epochs = 2
a = 0.004
b = 230
gamma = 0.55
test_accuracy_av = np.zeros((522, run_size))
#test_accuracy=[]
#iterlist = []

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


for run in range(0, run_size):
    total_idx = np.random.choice(total_num_samples, total_num_samples, replace=False)
    train_idx = total_idx[:train_num_samples]
    test_idx = total_idx[-test_num_samples:]

    # Train and test data sets
    train_data = a9aDataset(data=Xtr, labels=Ytr, train_idx=train_idx, test_idx=test_idx, train=True)
    test_data = a9aDataset(data=Xtr, labels=Ytr, train_idx=train_idx, test_idx=test_idx, train=False)

    # Train and test data loaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Model and loss
    model = a9a_model(num_inputs=num_feat).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    # Optimizer
    optimizer = sgld(model.parameters(), lr=0.01, weight_decay=True, num_batches=len(train_dataloader), addnoise=True)
    iterno = 1
    adjust_lr(optimizer, iterno=iterno, a=a, b=b, gamma=gamma)

    # Test accuracy before training
    test_accuracy = [accuracy_evaluation(model, test_dataloader, device)]
    iterlist = [iterno]

    # Training
    for epoch in range(1, num_epochs + 1):

        model.train()
    # tot_loss = 0.0
        for data, labels in train_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()
            iterno += 1
            adjust_lr(optimizer, iterno=iterno, a=a, b=b, gamma=gamma)

            if iterno % 10 == 0:
                acc = accuracy_evaluation(model, test_dataloader, device)
                print('Iteration: {}. Accuracy: {}'.format(iterno, acc))
                test_accuracy.append(acc)
                iterlist.append(iterno)
    test_accuracy_av[:, run] = test_accuracy
mean_accuracy = np.mean(test_accuracy_av, axis=1)
std_accuracy = np.std(test_accuracy_av, axis=1)


# Results
fontsize = 10
f1 = plt.figure()
ax = f1.gca()
plt.plot(np.asarray(iterlist), np.asarray(mean_accuracy))
ax.fill_between(np.asarray(iterlist), (np.asarray(mean_accuracy)-std_accuracy), (np.asarray(mean_accuracy)+std_accuracy), color='b', alpha=.3)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
plt.show()

