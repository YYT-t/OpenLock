import torch.nn as nn

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        #  b, 2, 3, 3
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),   # 16, 3, 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),  # 32, 3, 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)  # 64, 3, 3
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=64*3*3, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.feature(x)
        # print("x.size()=", x.size())
        x = x.view(x.size(0), -1)
        x = self.linear(x)  # b
        return x.squeeze()

# class Convnet(nn.Module):
#     def __init__(self):
#         super(Convnet, self).__init__()
#         #1, 3, 3
#         self.feature = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_features=4),
#             nn.ReLU(inplace=True),   # 4, 3, 3
#             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_features=8),
#             nn.ReLU(inplace=True)  # 8, 3, 3
#         )
#         self.linear = nn.Sequential(
#             nn.Linear(in_features=8*3*3, out_features=128),
#             nn.BatchNorm1d(num_features=128),
#             nn.ReLU(True),
#             nn.Linear(in_features=128, out_features=64),
#             nn.BatchNorm1d(num_features=64),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(64, 9)
#         )
#     def forward(self, x):
#         x = self.feature(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)  # b, 9
#         return x
