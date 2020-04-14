import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
from torch.utils import data
import numpy as np
import pickle

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/MPNet/dataset/'


# npd = []
#
# for file in glob.glob(ws_location+'points*.npy'):
#     narray = np.load(file)
#     npd.append(narray)

with open(ws_location+'pickle_test.pickle', 'rb+') as depickle_file:
    npd = pickle.load(depickle_file)
depickle_file.close()

print("files extracted")
train_data = npd[:900]
test_data = npd[900:]

batch_size = 16
num_epochs = 100
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size,
                                           shuffle=True)

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(115200, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 32))

        self.decoder = nn.Sequential(nn.Linear(32, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 115200))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

mse_loss = nn.MSELoss()

def combined_loss_function(W, x, h, recons_x, lam):
    mse = mse_loss(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)

autoencoder = CAE()

if(torch.cuda.is_available()):
    autoencoder.cuda()

parameters = autoencoder.parameters()
optimizer = torch.optim.SGD(parameters, lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, points in enumerate(train_loader):
        points = points.float()
        points = Variable(points)

        # if(torch.cuda.is_available()):
        points = points.cuda()
        encoded, decoded = autoencoder(points)
        W = autoencoder.state_dict()['encoder.4.weight']
        loss = combined_loss_function(W,points,encoded,decoded,0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((i + 1) % 5 == 0):
            print('epoch {0}/{1}, step {2}/{3}, loss = {4:4f}'.format(epoch + 1, num_epochs, i + 1, n_total_steps,
                                                                      loss.item()))

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for points in test_loader:
        points = points.float()
        if(torch.cuda.is_available()):
            points = points.cuda()
        _, decoded = autoencoder(points)
        n_samples += labels.shape[0]
        n_correct = (decoded == points).sum().item()

    acc = 100.0 * n_correct / n_samples
    print('accuracy = {0}'.format(acc))