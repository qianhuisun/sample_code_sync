import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)
        #x = torch.flatten(x, 1) # dim = 0 is the image index in batch, dim = 1 refers to within an image
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #return F.log_softmax(x, dim = 1)
        return F.softmax(x, dim = 1)


training_data = np.load("training_data.npy", allow_pickle = True)
print(len(training_data))

#print(training_data[0])
#plt.imshow(training_data[0][0], cmap = "gray")
#plt.show()

net = Net()

optimizer = optim.Adam(net.parameters(), lr = 0.001)
# loss_function = F.nll_loss()
loss_function = nn.MSELoss()

IMG_SIZE = 54

x = torch.Tensor([i[0] for i in training_data]).view(-1, IMG_SIZE, IMG_SIZE)
x /= 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1

val_size = int(len(x) * VAL_PCT)

print(val_size)

train_x = x[:-val_size]
train_y = y[:-val_size]

test_x = x[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 5

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_x), BATCH_SIZE)):

        batch_x = train_x[i:i+BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        
        output = net(batch_x)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()

print("loss = ", loss)

correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_x))):
        real_class = torch.argmax(test_y[i])
        output = net(test_x[i].view(-1, 1, IMG_SIZE, IMG_SIZE))[0]
        predicted_class = torch.argmax(output)
        if predicted_class == real_class:
            correct += 1

        total += 1

print("Accuracy = ", round(correct / total, 3))