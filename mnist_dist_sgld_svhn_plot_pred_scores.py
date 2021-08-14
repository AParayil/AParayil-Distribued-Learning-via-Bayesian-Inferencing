import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from models.LeNet5 import LeNet5
from optimizers.dist_SGLD import dsgld
import matplotlib.pyplot as plt
from utils.getlaplacian import getlaplacian
import numpy as np
import math
import copy
import seaborn as sns
from sklearn import metrics


def adjust_lr(optimizer, modelparams, iterno, alpha0, beta0, gamma):
    alpha = alpha0 / ((1 + gamma * iterno) ** 0.55)
    beta = beta0 / ((1 + gamma * iterno) ** 0.05)
    for param_group in optimizer.param_groups:
        param_group['alpha'] = alpha
        param_group['beta'] = beta
        param_group['allmodels'] = modelparams


def loss_function_evaluation(net, data_loader, criterion, device):
    loss = 0.0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        loss += criterion(outputs, labels).cpu().item()

    return loss / len(data_loader)


def accuracy_evaluation(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)

        outputs = net(images)

        predicted = outputs.argmax(dim=1, keepdim=True)

        correct += predicted.cpu().eq(labels.view_as(predicted.cpu())).sum().item()

        total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


def get_scores(net, data_loader, device):
    net.eval()

    pred_prob = np.zeros((len(data_loader.dataset), 10))
    for i, (images, _) in enumerate(data_loader):

        images = images.to(device)

        outputs = net(images)

        pred_prob[i*data_loader.batch_size: i*data_loader.batch_size + images.shape[0], :] = F.softmax(outputs, dim=1).detach().cpu().numpy()

    return pred_prob


train_batch_size = 256
test_batch_size = 4096
num_epochs = 10
num_agents = 5
torch.manual_seed(1)
gamma = 1 / 230.0
alpha0 = 16.0 * 1e-4 / ((230 ** 0.55) * num_agents)
adj, lap = getlaplacian(num_agents, type=0)
_, sv, _ = np.linalg.svd(lap)
beta0 = 1.01 / np.max(sv)

mnist_ind = [2266, 6572, 1678, 6651, 717, 7902, 4360, 6576, 1378, 1178, 3775, 2752, 8754, 2373, 3610]
svhn_ind = [9278, 9572, 16174, 10425, 5854, 7498, 3490, 10421, 17577, 5860, 14768, 1301, 15322, 25763, 8407]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train and test data sets
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = dsets.MNIST(root='./data', train=True, transform=trans, download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=trans)

num_datasamples = int(len(train_dataset) / num_agents)
lengths = [num_datasamples] * num_agents
num_mini_batches = math.ceil(num_datasamples / train_batch_size)

# Train and test data loaders
train_data_split = torch.utils.data.random_split(dataset=train_dataset, lengths=lengths)
train_loader_list = []
for k in range(num_agents):
    train_loader_list.append(torch.utils.data.DataLoader(dataset=train_data_split[k], batch_size=train_batch_size,
                                                         shuffle=True))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

svhn_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 32)), transforms.ToTensor()])
test_svhn_loader = torch.utils.data.DataLoader(dsets.SVHN('./data', split='test', download=True, transform=svhn_trans), batch_size=test_batch_size, shuffle=False)

vis_mnist_images = torch.zeros(len(mnist_ind), 1, 32, 32)
vis_mnist_labels = torch.zeros(len(mnist_ind), dtype=torch.long)
vis_svhn_images = torch.zeros(len(svhn_ind), 1, 32, 32)
vis_svhn_labels = torch.zeros(len(svhn_ind), dtype=torch.long)

# Model and loss
model = []
modelparams = []
for k in range(num_agents):
    net = LeNet5().to(device)
    model.append(net)
    modelparams.append(copy.deepcopy(net))

criterion = nn.CrossEntropyLoss(reduction='sum')

# Optimizer
optimizerlist = []
iterno = 1
for k in range(num_agents):
    optimizer = dsgld(model[k].parameters(), allmodels=modelparams, adj_vec=adj[k, :], alpha=alpha0, beta=beta0,
                      norm_sigma=0.0, num_batches=len(train_loader_list[k]), addnoise=True)
    adjust_lr(optimizer, modelparams, iterno=iterno, alpha0=alpha0, beta0=beta0, gamma=gamma)
    optimizerlist.append(optimizer)

# Test accuracy before training
net_test_accuracy = []
for k in range(num_agents):
    net_test_accuracy.append(accuracy_evaluation(model[k], test_loader, device))
test_accuracy = [net_test_accuracy]
iterlist = [iterno]

class_scores_svhn = []
class_scores_mnist = []
for epoch in range(num_epochs):

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
            outputs = cur_model(data)
            loss = criterion(outputs, labels)
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
                net_test_accuracy.append(accuracy_evaluation(model[k], test_loader, device))
            test_accuracy.append(net_test_accuracy)
            print('Iteration: {}. Accuracy: {}'.format(iterno, np.asarray(net_test_accuracy)))
            iterlist.append(iterno)

        if epoch>4:
            agent_class_scores_svhn = []
            agent_class_scores_mnist = []
            for k in range(num_agents):
                agent_class_scores_svhn.append(get_scores(model[k], test_svhn_loader, device))
                agent_class_scores_mnist.append(get_scores(model[k], test_loader, device))

            class_scores_svhn.append(agent_class_scores_svhn)
            class_scores_mnist.append(agent_class_scores_mnist)



class_pred_prob_svhn = np.asarray(class_scores_svhn)
class_pred_prob_mnist = np.asarray(class_scores_mnist)
vis_mnist_images = vis_mnist_images.cpu().detach().numpy().squeeze()
vis_svhn_images = vis_svhn_images.cpu().detach().numpy().squeeze()

# plot predicted and actual labels for selected indices and  corresponding  pred_probability for one agent
f1 = plt.figure()
k = 0

exp_class_pred_prob_mnist = np.mean(class_pred_prob_mnist[:, k, :, :].squeeze(), axis=0)
std_class_pred_prob_mnist = np.std(class_pred_prob_mnist[:, k, :, :].squeeze(), axis=0)
exp_max_pred_prob_mnist = np.max(exp_class_pred_prob_mnist, axis=1)
mnist_predicted = np.argmax(exp_class_pred_prob_mnist, axis=1)
mnist_labels = test_dataset.targets.numpy()

exp_class_pred_prob_svhn = np.mean(class_pred_prob_svhn[:, k, :, :].squeeze(), axis=0)
std_class_pred_prob_svhn = np.std(class_pred_prob_svhn[:, k, :, :].squeeze(), axis=0)
exp_max_pred_prob_svhn = np.max(exp_class_pred_prob_svhn, axis=1)
svhn_predicted = np.argmax(exp_class_pred_prob_svhn, axis=1)

for ind in range(len(mnist_ind)):
    mnist_pr_label = mnist_predicted[mnist_ind[ind]]
    plt.subplot(5, 3, ind+1)
    plt.axis('off')
    plt.imshow(vis_mnist_images[ind, :, :], cmap='gray')
    plt.title('{:.0f},'.format(mnist_pr_label) + ' '+'{:.1f}'.format(exp_class_pred_prob_mnist[mnist_ind[ind], mnist_pr_label]*100)+'%')
f1.savefig("testsvhn.png")

# plot actual and predicted labels across whole test data as a  heat map

cm = metrics.confusion_matrix(y_true=mnist_labels, y_pred=mnist_predicted, labels=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
f2 = plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True,
            linewidths=.5, square=True, cmap='Blues_r')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# print mean and svd of the prediction for selected indices for all agents
indx = 0
for k in range(num_agents):

    exp_class_pred_prob_mnist = np.mean(class_pred_prob_mnist[:, k, :, :].squeeze(), axis=0)
    std_class_pred_prob_mnist = np.std(class_pred_prob_mnist[:, k, :, :].squeeze(), axis=0)
    exp_max_pred_prob_mnist = np.max(exp_class_pred_prob_mnist, axis=1)
    mnist_predicted = np.argmax(exp_class_pred_prob_mnist, axis=1)



    exp_class_pred_prob_svhn = np.mean(class_pred_prob_svhn[:, k, :, :].squeeze(), axis=0)
    std_class_pred_prob_svhn = np.std(class_pred_prob_svhn[:, k, :, :].squeeze(), axis=0)
    exp_max_pred_prob_svhn = np.max(exp_class_pred_prob_svhn, axis=1)
    svhn_predicted = np.argmax(exp_class_pred_prob_svhn, axis=1)

    mnist_pr_label = mnist_predicted[mnist_ind[indx]]
    svhn_pr_label = svhn_predicted[svhn_ind[indx]]

    print(k)
    print(mnist_ind[indx])
    print(exp_class_pred_prob_mnist[mnist_ind[indx], mnist_pr_label])
    print(std_class_pred_prob_mnist[mnist_ind[indx], mnist_pr_label])
    print(svhn_ind[indx])
    print(exp_class_pred_prob_svhn[svhn_ind[indx], svhn_pr_label])
    print(std_class_pred_prob_svhn[svhn_ind[indx], svhn_pr_label])
