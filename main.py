from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input = nn.Linear(28*28, 512)
        self.func = nn.ReLU()
        self.hidden = nn.Linear(512, 512)
        self.func_h = nn.ReLU()
        self.output = nn.Linear(512, 10)

        self.sequence = [self.flatten, self.input, self.func, self.hidden, self.func_h, self.output]


    def forward(self, x):

        output = x

        for layer in self.sequence:
            output = layer(output)

        return output
    
def train(model, data, loss, optimiser):

    model.train()
    for batch_index, (X, y) in enumerate(data):

        X, y = X.to(device), y.to(device)
        outputs = model(X)
        error = loss(outputs, y)

        error.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch_index % 100 == 0:
            print(f"Loss: {error.item():>7f}")


def test(test, model, loss):

    size = len(test.dataset)
    num_batches = len(test)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test:

            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size

    print(f"Error: {correct*100:>0.1f}% Avg Loss: {test_loss:>8f} \n")

def get_data(num_batches):
    transform = transforms.ToTensor()
    #Downloads only once
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train = DataLoader(train_data, batch_size=num_batches)
    test = DataLoader(test_data, batch_size=num_batches)

    return train, test


if __name__ == "__main__":

    train_data, test_data = get_data(64)
    model = NeuralNet().to(device)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        print(f"Epoch: {epoch+1}")
        train(model, train_data, loss, optim)
        test(test_data, model, loss)
    print("Done!")
