import torch
import os
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.nn.modules.module import _addindent
import numpy as np
import re

from scipy.interpolate import interp1d
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
)
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
#from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm


epochs = 1000
no_cuda = False
seed = 1
log_interval = 50


BATCH_SIZE = 256
FRAMESIZE = 512
OVERLAP = 256
FFTSIZE = 512
RATE = 16000
FRAMEWIDTH = 2
FBIN = FRAMESIZE // 2 + 1
max_pad_len = 240
dimension = 200


cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")
print(device)

kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}


def natural_sort_key(s):
    return int("".join(re.findall("\d*", s)))


trainTransforms = transforms.Compose(
    [
        # resize
        transforms.Resize(dimension),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(hue=0.05, saturation=0.05),
        # to-tensor
        transforms.ToTensor(),
        # normalize
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
testTransforms = transforms.Compose(
    [
        # resize
        transforms.Resize(dimension),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

trainDataset = datasets.ImageFolder(root="Dataset/TRAIN/", transform=trainTransforms)

testDatasetReal = datasets.ImageFolder(
    root="Dataset/TEST/REAL/", transform=testTransforms
)
testDatasetFake = datasets.ImageFolder(
    root="Dataset/TEST/FAKE/", transform=testTransforms
)


def mse_loss_cal(input, target, avg_batch=True):

    ret = torch.mean((input - target) ** 2)

    return ret.item()


class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(0.2)
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(25 * 25 * 32, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)

        # Sampling vector
        self.fc3 = nn.Linear(1024, 1024)
        self.fc_bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 25 * 25 * 32)
        self.fc_bn4 = nn.BatchNorm1d(25 * 25 * 32)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(
            32, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))

        conv4 = conv4.view(-1, 25 * 25 * 32)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.50).exp_()
            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        fc4 = fc4.view(-1, 32, 25, 25)
        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        conv8 = self.conv8(conv7)
        return conv8.view(-1, 3, dimension, dimension)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


model = VAE_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_mse = customLoss()

val_losses_fake = []
val_losses_real = []
train_losses = []

trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testDataLoaderReal = DataLoader(testDatasetReal, batch_size=1)
testDataLoaderFake = DataLoader(testDatasetFake, batch_size=1)


print("[INFO] training dataset contains {} samples...".format(len(trainDataset)))
print("[INFO] test Real dataset contains {} samples...".format(len(testDataLoaderReal)))
print("[INFO] test Fake dataset contains {} samples...".format(len(testDataLoaderFake)))


def train(epoch):
    model.train()
    # print(torch_summarize(model))
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainDataLoader):
        data = data.to(device)
        # permute = [2, 1, 0]
        # data = data[:, permute, :, :]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(trainDataLoader.dataset),
                    128.0 * batch_idx / len(trainDataLoader),
                    loss.item() / len(data),
                )
            )
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(trainDataLoader.dataset)
        )
    )
    train_losses.append(train_loss / len(trainDataLoader.dataset))


# print(summary(model, (1, 64, 80)))

accur = 0.0
for epoch in range(1, epochs + 1):
    train(epoch)
    # test(epoch)

    loader1 = tqdm(testDataLoaderReal)
    loader2 = tqdm(testDataLoaderFake)
    sum1 = []
    sum2 = []

    with torch.no_grad():
        for i, (img, label) in enumerate(loader1):  # real
            model.zero_grad()
            # permute = [2, 1, 0]
            # img = img[:, permute, :, :]
            img = img.to(device)
            model.eval()
            recon_batch, mu, logvar = model(img)
            sum1.append(mse_loss_cal(recon_batch, img))
            # sum1.append(loss_mse(recon_batch, img, mu, logvar).item())

    with torch.no_grad():
        for i, (img, label) in enumerate(loader2):  # FAKE
            model.zero_grad()
            # permute = [2, 1, 0]
            # img = img[:, permute, :, :]
            img = img.to(device)
            model.eval()
            recon_batch, mu, logvar = model(img)
            sum2.append(mse_loss_cal(recon_batch, img))

        classSize = len(testDataLoaderReal)
        y_test = ([0] * classSize) + ([1] * classSize)

        y_test = np.asarray(y_test)

        y_score = np.asarray(sum1 + sum2)
        indices = np.arange(classSize + classSize)
        y_hat = y_score.copy()
        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1.0)

        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        thresh = interp1d(fpr, thresholds)(eer)

        y_hat[y_hat <= thresh] = 0.0
        y_hat[y_hat > thresh] = 1.0

        y_pp_test = np.asarray(y_test).reshape([-1, 1])
        print(classification_report(y_test, y_hat, digits=4))
        print(confusion_matrix(y_test, y_hat))
        print(" #################### ")
        cm = confusion_matrix(y_test, y_hat)
        recall = cm[0][0] / (cm[0][0] + cm[0][1])
        fallout = cm[1][0] / (cm[1][0] + cm[1][1])
        far = fallout
        frr = 1 - recall
        print("FPR=FAR", fallout)
        print("FNR=FRR", 1 - recall)
        print("HTER", (far + frr) / 2)
        print("EER", eer)
        print("thresh", thresh)

        a = len(np.array(sum1)[np.array(sum1) <= thresh] * dimension) / classSize
        b = len(np.array(sum2)[np.array(sum2) > thresh] * dimension) / classSize
        print(a)
        print(b)
        acc = (a + b) / 2
        print("Saving Model: " + str(epoch) + " Accuracy:" + str(acc))
        torch.save(
            model.state_dict(), "checkpoints/vae_pytorch_hand_" + str(epoch) + ".pt"
        )

# torch.save(model.state_dict(),
#           "dfdc/vae_pytorch_dfdc.pt")

# plt.figure(figsize=(15, 10))
# plt.plot(range(len(train_losses[1:])), train_losses[1:], c="dodgerblue")
# plt.plot(range(len(val_losses_fake[1:])), val_losses_fake[1:], c="tomato")
# plt.plot(range(len(val_losses_real[1:])), val_losses_real[1:], c="limegreen")
# plt.title("Loss per epoch", fontsize=18)
# plt.xlabel("epoch", fontsize=18)
# plt.ylabel("loss", fontsize=18)
# plt.legend(["Train. Loss", "Val. Loss (Fake)", "Val. Loss (Real)"], fontsize=18)
