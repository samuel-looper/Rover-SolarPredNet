import torch
from torchsummary import summary


# SolarPredConv.py: Defines a fully-convolutional model serving as a baseline Neural Network model
# to compare to SolarPredHybrid


class SolarPredConvNet(torch.nn.Module):
    # Neural Network model that directly predicts solar energy generation from navigation imagery

    # Inputs:   Pre-processed image (1160x240)
    # Outputs:  Predicted solar power generation (W)

    def __init__(self):
        super(SolarPredConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, stride=2, kernel_size=(7, 7), padding=2, padding_mode="reflect")
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.LeakyReLU()

        self.conv2 = torch.nn.Conv2d(16, 16, stride=2, kernel_size=(5, 5), padding=2, padding_mode="reflect")
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.LeakyReLU()

        self.conv3 = torch.nn.Conv2d(16, 32, stride=2, kernel_size=(3, 3), padding=2, padding_mode="reflect")
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout2d(p=0.5)

        self.conv4 = torch.nn.Conv2d(32, 64, stride=2, kernel_size=(3, 3), padding=0, padding_mode="reflect")
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.LeakyReLU()

        self.conv5 = torch.nn.Conv2d(64, 64, stride=4, kernel_size=(3, 3), padding=0, padding_mode="reflect")
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.LeakyReLU()
        self.dropout5 = torch.nn.Dropout2d(p=0.5)

        self.conv6 = torch.nn.Conv2d(64, 64, stride=1, kernel_size=(4, 1), padding=0, padding_mode="reflect")
        self.bn6 = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.LeakyReLU()
        self.dropout6 = torch.nn.Dropout2d(p=0.5)

        self.fc1 = torch.nn.Linear(1152, 240)
        self.relu7 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(240, 1)

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        x3 = self.dropout3(self.relu3(self.bn3(self.conv3(x2))))
        x4 = self.relu4(self.bn4(self.conv4(x3)))
        x5 = self.dropout5(self.relu5(self.bn5(self.conv5(x4))))
        x6 = self.relu6(self.bn6(self.conv6(x5)))

        x_flat = torch.flatten(x6, 1, 3)
        x7 = self.relu7(self.fc1(self.dropout6(x_flat)))
        x8 = self.fc2(x7)

        return x8


if __name__ == "__main__":
    solarpred_cnn = SolarPredConvNet()
    summary(solarpred_cnn, input_size=(3, 240, 1160))
