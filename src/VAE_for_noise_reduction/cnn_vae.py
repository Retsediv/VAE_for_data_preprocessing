import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, ZDIMS):
        super(VAE, self).__init__()

        self.ZDIMS = ZDIMS

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.fc1 = nn.Linear(784, 400)
        # rectified linear unit layer from 400 to 400
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(400, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections

        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(ZDIMS, 400)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
        h1 = self.relu(self.fc1(x))  # type: Variable
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)

        else:
            return mu

    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        x = x.float()
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def transform(self, x, eps=0.1):
        mean, var = self.encode(x)
        noise = Variable(1. + torch.randn(x.size(0), self.ZDIMS) * eps).cuda()
        z = self.reparameterize(mean * noise, var * noise)
        out = self.decode(z)

        return out


def loss_function(recon_x, x, mu, logvar, BATCH_SIZE) -> Variable:
    x = x.view(-1, 784)
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 784

    return BCE + KLD
