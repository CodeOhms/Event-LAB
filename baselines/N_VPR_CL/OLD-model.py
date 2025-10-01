from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

# from N_VPR_CL.SimCLR.simclr.builder import SimCLR

# class SynSSparkNN(nn.Module):
#     def __init__(self, num_classes=1024):
#         super(SynSSparkNN, self).__init__()

#         self.seq_layers = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("conv1", nn.Conv2d(3, 128, 3, padding=1)),
#                     ("actv1", nn.ReLU()),
#                     ("conv2", nn.Conv2d(128, 256, 3, padding=1)),
#                     ("actv2", nn.ReLU()),
#                     ("maxpl1", nn.MaxPool2d((2,2))),
#                     ("conv3", nn.Conv2d(256, 512, 3, padding=1)),
#                     ("actv3", nn.ReLU()),
#                     ("maxpl1", nn.MaxPool2d((2,2))),
#                     ("flatten", nn.Flatten()),
#                     ("linear", nn.LazyLinear(1024)),
#                     ("actv4", nn.ReLU())
#                 ]
#             )
#         )
#         self.fc = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.seq_layers(x)
#         x = self.fc(x)
#         return x


class Test(nn.Module):
    def __init__(self, num_classes):
        super(Test, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.linear1 = nn.Linear(16, 32, bias=False)
        self.fc = nn.Linear(32, num_classes)
        self.actv1 = nn.ReLU()
        self.actv2 = nn.ReLU()
        self.actv3 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        # x = nn.ReLU()(x)
        x = self.actv1(x)
        x = self.conv2(x)
        x = self.actv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.actv3(x)
        x = self.fc(x)

        return x


class SynSSparkNN(Test):
    def __init__(self, num_classes):
        super().__init__(num_classes)


# class SimCLR_SynSSpark(SimCLR):
#     def __init__(self, dim=128, mlp_dim=1024, T=0.1):
#         # Custom base encoder for the SynSense Spark chip
#         base_encoder = partial(SynSSparkNN)
#         super(SimCLR_SynSSpark, self).__init__(base_encoder, dim, mlp_dim, T)

#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         hidden_dim = self.encoder.fc.weight.shape[1]

#         # projectors
#         self.encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
