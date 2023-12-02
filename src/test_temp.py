import torch.nn as nn
import torch
from torchvision import models
from pathlib import Path
from glob import glob
import numpy as np

from models.model_unet1 import UNetClassifier1, load_unet1_with_classifier_weights
from models.model_unet2 import UNetClassifier2, load_unet2_with_classifier_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetClassifier1({'num_out':2})
model = load_unet1_with_classifier_weights({'num_out':2})

#model = UNetClassifier2({'num_out':2})
#model = load_unet2_with_classifier_weights({'num_out':2})

for param in model.parameters():
    print(param.requires_grad)

image = np.random.rand(1,1,256,256).astype(float)
image = torch.tensor(image).float()
model.eval()

print(model)
with torch.no_grad():
    output = model(image)
    for param in model.parameters():
        print(param.requires_grad)





