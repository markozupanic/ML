from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt

IMAGE_INPUT_SIZE=(28,28)

device="cuda" if torch.cuda.is_available() else "cpu"
image_path="2.png"
image = Image.open(image_path).convert('L')#L pretvara u gray scale

model=torch.load("custom_nn/models/model_after_epoch_6.pt").to(device)  #najbolji model
model.eval()


preprocess=transforms.Compose([transforms.Resize(IMAGE_INPUT_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=0.5,std=[0.5])])

input_tensor=preprocess(image)
input_tensor=input_tensor.to(device)
torch.unsqueeze(input_tensor, 0)
output=model(input_tensor)

output=nn.functional.softmax(output,1)

predicted_class=output.argmax(1).item()

probability=output[0][predicted_class]

plt.imshow(image,cmap='gray')
plt.title("Class: {}\nProbability: {:.4f}".format(predicted_class,probability))
plt.show()
















