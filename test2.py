import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader= DataLoader(train_data, batch_size=64, shuffle=True)
test_loader= DataLoader(test_data, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(28*28,10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        return x
    
model= SimpleNN()

criterion= nn.CrossEntropyLoss()
optimizer= optim.SGD(model.parameters(), lr=0.001)

epochs=3
for epochs in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs= model(inputs)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
           print(f'Epoch {epochs+1}/{epochs} Step {i+1}/{len(train_loader)} Loss {loss.item()}')

correct = 0 
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs= model(images)
        _, predicted= torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'테스트 정확도: {100*correct/total}')