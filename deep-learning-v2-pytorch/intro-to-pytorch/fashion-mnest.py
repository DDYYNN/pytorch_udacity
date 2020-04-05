import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

## To show one of the images : uncomment this
# image, label = next(iter(trainloader))
# helper.imshow(image[0,:]);


##############################################################################################


# TODO: Define your network architecture here
from torch import nn, optim

# Hyperparameters
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
f_nn = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                     nn.ReLU(),
                     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                     nn.ReLU(),
                     nn.Linear(hidden_sizes[1], output_size),
                     nn.LogSoftmax(dim=1))


# TODO: Create the network, define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(f_nn.parameters(), lr=0.005)


# TODO: Train the network here
epoch = 10

for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector,
        # automatically catching batch size.
        images = images.view(images.shape[0], -1)
        
        # Initialize gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = f_nn.forward(images)
        loss = criterion(output, labels)
        
        # Backward propagation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import helper

# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# TODO: Calculate the class probabilities (softmax) for img
# turn off gradient b/c we don't need it
with torch.no_grad():
    logps = f_nn(img)

# probability = exp(log-probability)
ps = torch.exp(logps)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
