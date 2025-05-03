from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
from utils import load_vision_model, resnet18_cifar10_modifier, get_cifar10_test_loader
import torch

model = load_vision_model(
    "resnet18",
    pretrained_path="resnet18.pth",
    modifier_before_load=resnet18_cifar10_modifier,
    modifier_after_load=None,
).to("cuda")

# eval before
model.eval()
cifar = get_cifar10_test_loader(batch_size=512)
correct = 0
total = 0
with torch.no_grad():
    for data in cifar:
        images, labels = data
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total} % before quantization")


quantize_(model, int8_dynamic_activation_int8_weight())

print(model)
correct = 0
total = 0
with torch.no_grad():
    for data in cifar:
        images, labels = data
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total} % after quantization")