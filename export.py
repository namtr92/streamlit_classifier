import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets, transforms, models

model_name = 'deit_tiny'

import timm
model = timm.create_model('deit_tiny_patch16_224',pretrained=False,num_classes=16)
model.load_state_dict(torch.load('deploy/deit_tiny_pad.pth'))

#model = models.resnet18(pretrained=False)
#model.fc = nn.Linear(512, num_classes)  # Modify num_classes based on your dataset
#model = mobileone(variant='s0',num_classes=num_classes)
#model.load_state_dict(torch.load(f'models/{model_name}_pad.pth'))
#torch.save(model,f'models/{model_name}_pad_full.pth')
save_dir = f'models/{model_name}_pad.pth'
torch.onnx.export(model,torch.randn(1,3,224,224),save_dir.replace('.pth','.onnx'),opset_version=15)