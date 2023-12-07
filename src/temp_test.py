import torch.nn as nn
import torch
from torchvision import models
from pathlib import Path
from glob import glob
import numpy as np

from models.model_unet1 import UNetClassifier1, load_unet1_with_classifier_weights
from models.model_unet2 import UNetClassifier2, load_unet2_with_classifier_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = UNetClassifier1({'num_out':2})
#model = load_unet1_with_classifier_weights({'num_out':2})

model = UNetClassifier2({'num_out':2})
model = load_unet2_with_classifier_weights({'num_out':2})


model = models.resnet18()

with torch.set_grad_enabled(True):
    for param in model.parameters():
        print(param.requires_grad)

print(model)
exit()

image = np.random.rand(1,1,224,224).astype(float)
image = torch.tensor(image).float()
model.eval()

with torch.no_grad():
    output = model(image)



    # run 4: 65 epochs, lr: 1e-4, step_size=10, gamma=0.7
    # run 5: 16 epochs, lr_ 1e-4, step_size=5, gamma=0.2
