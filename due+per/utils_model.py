import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__device = device
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(64*7*7, 512)
        
        self.__advantage_layer = nn.Linear(512, action_dim)
        # value layer
        self.__value_layer = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        adv = F.relu(self.__fc1(x.view(x.size(0), -1)))
        val = F.relu(self.__fc2(x.view(x.size(0), -1)))
        
        advantage = self.__advantage_layer(adv)
        value = self.__value_layer(val)
        q = value + advantage -  advantage.mean(dim=-1, keepdim=True)
        return q

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
