import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np, random

# Custom dataset for paired MNIST
class PairMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        self.transform = transforms.ToTensor()
        self.index_by_label = {}
        for i, label in enumerate(self.targets):
            label = int(label)
            self.index_by_label.setdefault(label, []).append(i)
        self.pairs, self.labels = [], []
        for idx in range(len(self.data)):
            img1_label = int(self.targets[idx])
            # Create a positive pair
            pos_idx = random.choice(self.index_by_label[img1_label])
            self.pairs.append((idx, pos_idx))
            self.labels.append(1)
            # Create a negative pair
            neg_label = random.choice([l for l in self.index_by_label if l != img1_label])
            neg_idx = random.choice(self.index_by_label[neg_label])
            self.pairs.append((idx, neg_idx))
            self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2 = self.pairs[idx]
        img1 = self.transform(self.data[i1].numpy())
        img2 = self.transform(self.data[i2].numpy())
        # Stack images to form a 2-channel tensor
        pair = torch.cat([img1, img2], dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pair, label

# Feature extractor network
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Siamese network using the shared feature extractor
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, 2, 28, 28)
        img1 = x[:, 0:1, :, :]
        img2 = x[:, 1:2, :, :]
        feat1 = self.feature(img1)
        feat2 = self.feature(img2)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.classifier(combined)

# Prepare datasets (using 10% of MNIST)
train_full = MNIST(root='./data', train=True, download=True)
test_full  = MNIST(root='./data', train=False, download=True)
train_subset = torch.utils.data.Subset(train_full, np.random.choice(len(train_full), int(0.1 * len(train_full)), replace=False))
test_subset  = torch.utils.data.Subset(test_full,  np.random.choice(len(test_full),  int(0.1 * len(test_full)),  replace=False))

train_dataset = PairMNIST(train_subset)
test_dataset  = PairMNIST(test_subset)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f'Test Accuracy: {100 * correct/total:.2f}%')