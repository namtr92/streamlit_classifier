import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets, transforms, models
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

from mobileone import mobileone
quant_nn.TensorQuantizer.use_fb_fake_quant = True

from tqdm import tqdm
quant_modules.initialize()

model_name = 'resnet18'

def compute_amax(model, **kwargs):
     # Load calib result
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 if isinstance(module._calibrator, calib.MaxCalibrator):
                     module.load_calib_amax(strict=False)
                 else:
                     module.load_calib_amax(**kwargs,strict=False)
             print(F"{name:40}: {module}")
     model.cuda()
def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
         model(image.cuda())
         if i >= num_batches:
             break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()

# Define transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(128,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Modify the root directory based on your dataset structure
dataset_path = r'D:\images\kcc\dataset\classifier'
custom_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
duplicated_dataset = ConcatDataset([custom_dataset] * 2)
# Define DataLoader
batch_size = 2
data_loader = DataLoader(duplicated_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
num_classes = len(custom_dataset.classes)

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, num_classes)  # Modify num_classes based on your dataset
#model = mobileone(variant='s0',num_classes=num_classes)
model.load_state_dict(torch.load(f'models/{model_name}_pad.pth'))
#torch.save(model,f'models/{model_name}_pad_full.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, data_loader, num_batches=100)
    compute_amax(model, method="percentile", percentile=99.99)

from tqdm import tqdm
save_dir = f'models/{model_name}_pad_int8.pth'
model.load_state_dict(torch.load(save_dir)) 
num_epochs = 2
correct_predictions = 0
total_samples = 0
max_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    data_loader_with_progress = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', dynamic_ncols=True)
    for inputs, labels in data_loader_with_progress:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        data_loader_with_progress.set_postfix({'Loss': loss.item()}, refresh=True)
    accuracy = correct_predictions / total_samples
    if (accuracy> max_accuracy):
        max_accuracy = accuracy
        torch.save(model.state_dict(),save_dir)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Acc: {accuracy}')
torch.onnx.export(model,torch.randn(4,3,128,128).to(device),save_dir.replace('.pth','.onnx'),opset_version=13)