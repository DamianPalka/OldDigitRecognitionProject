import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

def preprocessImage(img):
    trans = transforms.Compose([transforms.ToTensor()])
    img_t = trans(img)

    print(img_t.shape)

    batch_t = img_t.mean(0).view(1, -1).unsqueeze(0)

    print(batch_t.shape)

    return batch_t

learning_rate = 1e-3

model = nn.Sequential(nn.Linear(784, 30).to(device="cuda"), nn.Tanh(), nn.Linear(30, 10).to(device="cuda")).to(device="cuda")
model.load_state_dict(torch.load("model.pt"))

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