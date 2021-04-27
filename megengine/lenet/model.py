import megengine.module as M
import megengine.functional as F

class LeNet32x32(M.Module):
    def __init__(self):
        super().__init__()
        # single channel image, two 5x5 Conv + ReLU + Pool
        self.conv1 = M.Conv2d(1, 6, 5)
        self.relu1 = M.ReLU()
        self.pool1 = M.MaxPool2d(2, 2)
        self.conv2 = M.Conv2d(6, 16, 5)
        self.relu2 = M.ReLU()
        self.pool2 = M.MaxPool2d(2, 2)
        # two FC + ReLU
        self.fc1 = M.Linear(16 * 5 * 5, 120)
        self.relu3 = M.ReLU()
        self.fc2 = M.Linear(120, 84)
        self.relu4 = M.ReLU()
        # classifier
        self.classifier = M.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # F.flatten reshape the tensor x with (N, C, H, W) into shape of (N, C*H*W)
        # i.e., x = x.reshape(x.shape[0], -1)
        # x.shape: (256, 16, 5, 5)
        x = F.flatten(x, 1)
        # x.shape: (256, 400)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.classifier(x)
        return x

class LeNet224x224(M.Module):
    def __init__(self):
        super().__init__()
        # single channel image, two 5x5 Conv + ReLU + Pool
        self.conv1 = M.Conv2d(1, 6, 5)
        self.relu1 = M.ReLU()
        self.pool1 = M.MaxPool2d(2, 2)
        self.conv2 = M.Conv2d(6, 16, 5)
        self.relu2 = M.ReLU()
        self.pool2 = M.MaxPool2d(2, 2)
        # two FC + ReLU
        self.fc1 = M.Linear(16 * 53 * 53, 120)
        self.relu3 = M.ReLU()
        self.fc2 = M.Linear(120, 84)
        self.relu4 = M.ReLU()
        # classifier
        self.classifier = M.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # F.flatten(x, 1) reshape the tensor x with (N, C, H, W) along
        # the 1st dimension into shape of (N, C*H*W),
        # i.e., x = x.reshape(x.shape[0], -1)
        # x.shape: (256, 16, 53, 53)
        x = F.flatten(x, 1)
        # x.shape: (256, 16*53*53)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.classifier(x)
        return x
