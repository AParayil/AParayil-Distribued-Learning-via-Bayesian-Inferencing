import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from models.LeNet5 import LeNet5
from optimizers.SGLD import sgld
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def adjust_lr(optimizer, iterno, a, b, gamma):
    lr = a / ((b + iterno) ** gamma)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def attribute_image_features(algorithm, input, labels, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels,
                                              **kwargs
                                              )

    return tensor_attributions


train_batch_size = 1024
test_batch_size = 4096
num_epochs = 10
# gamma = 1e-5
# alpha0 = 1e-3
torch.manual_seed(1)
a = 3.5 * 1e-4 # 0.2
b = 230
gamma = 0.55

mnist_ind = [2266, 6572, 1678, 6651, 717, 7902, 4360, 6576, 1378, 1178, 3775, 2752, 8754, 2373, 3610]
svhn_ind = [9278, 9572, 16174, 10425, 5854, 7498, 3490, 10421, 17577, 5860, 14768, 1301, 15322, 25763, 8407]


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train and test data sets and loaders
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = dsets.MNIST(root='./data', train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=trans, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

svhn_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 32)), transforms.ToTensor()])
svhn_dataset = dsets.SVHN('./data', split='test', download=True, transform=svhn_trans)
test_svhn_loader = torch.utils.data.DataLoader(dataset=svhn_dataset, batch_size=test_batch_size, shuffle=False)

vis_mnist_images = torch.zeros(len(mnist_ind), 1, 32, 32)
vis_mnist_labels = torch.zeros(len(mnist_ind), dtype=torch.long)
vis_svhn_images = torch.zeros(len(svhn_ind), 1, 32, 32)
vis_svhn_labels = torch.zeros(len(svhn_ind), dtype=torch.long)
for i in range(len(mnist_ind)):
    ttmnist = test_dataset[mnist_ind[i]]
    ttsvhn = svhn_dataset[svhn_ind[i]]
    vis_mnist_images[i, :, :, :] = ttmnist[0]
    vis_mnist_labels[i] = ttmnist[1]
    vis_svhn_images[i, :, :, :] = ttsvhn[0]
    vis_svhn_labels[i] = ttsvhn[1]

# Model and loss
net = LeNet5().to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')

# Optimizer
optimizer = sgld(net.parameters(), lr=0.01, norm_sigma=0.0, num_batches=len(train_loader), addnoise=True) # Gradient of Kaiming uniform prior is 0
iterno = 1
adjust_lr(optimizer, iterno=iterno, a=a, b=b, gamma=gamma)

# Test accuracy before training
test_accuracy = [accuracy_evaluation(net, test_loader, device)]
iterlist = [iterno]
class_scores_svhn = []
class_scores_mnist = []
mnist_gt_attr_list = []
svhn_gt_attr_list = []
mnist_pr_attr_list = []
svhn_pr_attr_list = []
for epoch in range(num_epochs):

    for images, labels in train_loader:
        net.train()
        images = images.to(device)
        labels = labels.to(device)

        net.zero_grad()
        optimizer.zero_grad()

        outputs = net(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        iterno += 1
        adjust_lr(optimizer, iterno=iterno, a=a, b=b, gamma=gamma)

        if iterno % 10 == 0:
            acc = accuracy_evaluation(net, test_loader, device)
            print('Iteration: {}. Accuracy: {}'.format(iterno, acc))
            test_accuracy.append(acc)
            iterlist.append(iterno)

        if epoch > 4:
            net.eval()
            scores = get_scores(net, test_svhn_loader, device)
            class_scores_svhn.append(scores)
            scores = get_scores(net, test_loader, device)
            class_scores_mnist.append(scores)


class_pred_prob_svhn = np.asarray(class_scores_svhn)
exp_class_pred_prob_svhn = np.mean(class_pred_prob_svhn, axis=0)
std_class_pred_prob_svhn = np.std(class_pred_prob_svhn, axis=0)
exp_max_pred_prob_svhn = np.max(exp_class_pred_prob_svhn, axis=1)
weights_svhn = np.ones_like(exp_max_pred_prob_svhn) / len(exp_max_pred_prob_svhn)

class_pred_prob_mnist = np.asarray(class_scores_mnist)
exp_class_pred_prob_mnist = np.mean(class_pred_prob_mnist, axis=0)
std_class_pred_prob_mnist = np.std(class_pred_prob_mnist, axis=0)
exp_max_pred_prob_mnist = np.max(exp_class_pred_prob_mnist, axis=1)
weights_mnist = np.ones_like(exp_max_pred_prob_mnist) / len(exp_max_pred_prob_mnist)

mnist_predicted = np.argmax(exp_class_pred_prob_mnist, axis=1)
mnist_labels = test_dataset.targets.numpy()
test_mnist_acc = (mnist_predicted == mnist_labels).mean()

svhn_predicted = np.argmax(exp_class_pred_prob_svhn, axis=1)
svhn_labels = svhn_dataset.labels
test_svhn_acc = (svhn_predicted == svhn_labels).mean()

vis_mnist_images = vis_mnist_images.cpu().detach().numpy().squeeze()
vis_svhn_images = vis_svhn_images.cpu().detach().numpy().squeeze()

# plot predicted and actual labels for selected indices and  corresponding  pred_probability
f1 = plt.figure()
for ind in range(len(mnist_ind)):
    mnist_pr_label = mnist_predicted[mnist_ind[ind]]
    plt.subplot(5, 3, ind+1)
    plt.axis('off')
    plt.imshow(vis_mnist_images[ind, :, :], cmap='gray')
    plt.title('{:.0f},'.format(mnist_pr_label) + ' ' + '{:.1f}'.format(exp_class_pred_prob_mnist[mnist_ind[ind], mnist_pr_label]*100)+'%')
f1.savefig("testsvhn.png")
cm = metrics.confusion_matrix(y_true=mnist_labels, y_pred=mnist_predicted, labels=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


# plot actual and predicted labels across whole test data as a  heat map
f2 = plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True,
            linewidths=.5, square=True, cmap='Blues_r')

plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# print mean and svd of the prediction for selected indices
indx = 0
mnist_pr_label = mnist_predicted[mnist_ind[indx]]
svhn_pr_label = svhn_predicted[svhn_ind[indx]]
print(mnist_ind[indx])
print(vis_mnist_labels[indx])
print(mnist_pr_label)
print(exp_class_pred_prob_mnist[mnist_ind[indx], mnist_pr_label])
print(std_class_pred_prob_mnist[mnist_ind[indx], mnist_pr_label])
print(svhn_ind[indx])
print(vis_svhn_labels[indx])
print(svhn_pr_label)
print(exp_class_pred_prob_svhn[svhn_ind[indx], svhn_pr_label])
print(std_class_pred_prob_svhn[svhn_ind[indx], svhn_pr_label])


f3 = plt.figure()
plt.imshow(vis_mnist_images[indx, :, :], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.figure(1)
plt.imshow(vis_svhn_images[indx, :, :], cmap='gray')
plt.xticks([])
plt.yticks([])

f2.savefig("testsvhn1.png")
f3.savefig("testsvhn2.png")

