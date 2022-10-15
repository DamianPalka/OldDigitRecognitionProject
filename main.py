import torch

import torch.optim as optim
import torch.nn as nn

from torchvision import transforms
from torchvision import datasets
from PIL import Image

def preprocessImage(img):
    trans = transforms.Compose([transforms.ToTensor()])
    img_t = trans(img)

    print(img_t.shape)

    batch_t = img_t.mean(0).view(1, -1).unsqueeze(0)

    print(batch_t.shape)

    return batch_t

learning_rate = 1e-2

trainingData = datasets.MNIST("C:\\Users\\PC\\downloads", train=True, transform=transforms.ToTensor(), download=True)
valData = datasets.MNIST("C:\\Users\\PC\\downloads", train=False, transform=transforms.ToTensor(), download=True)

model = nn.Sequential(nn.Linear(784, 30).to(device="cuda"), nn.Tanh(), nn.Linear(30, 10).to(device="cuda")).to(device="cuda")

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

n_epochs = 200

trainingDataLoader = torch.utils.data.DataLoader(trainingData, batch_size=256, shuffle=True)
valDataLoader = torch.utils.data.DataLoader(valData, batch_size=256, shuffle=False)

for epoch in range(n_epochs):
    loss = 0

    for data, label in trainingDataLoader:
        data = data.to(device="cuda")
        label = label.to(device="cuda")
        batch_size = data.shape[0]
        prediction = model(data.view(batch_size, -1))

        loss = loss_fn(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 


    valLoss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in valDataLoader:
            data = data.to(device="cuda")
            label=label.to(device="cuda")
            batch_size = data.shape[0]
            prediction = model(data.view(batch_size, -1))
            valLoss = loss_fn(prediction, label)

            total += 1

    print("Epoch: " + str(epoch) + " training loss: " + str(loss) + " val loss: " + str(valLoss)) 

torch.save(model.state_dict(), "model.pt")

while (True):
    c = input("Press y to read digit, press n to exit")

    if (c == "n"):
        break

    img = Image.open("digit.jpg")

    batch_t = preprocessImage(img).to(device="cuda")

    output = model(batch_t)

    _, max = torch.max(output, dim=2)

    print(output)
    print("You drew a:\t"+ str(max[0, 0].item()))

    img.close()



       




