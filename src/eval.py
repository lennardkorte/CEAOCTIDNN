
from pathlib import Path
import torch
import numpy as np
import math
import os
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage

from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve

class Eval():
    def __init__(self, dataloader, device, model, loss_function, config, save_path_cv):
        model.eval()
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.squeeze().type(torch.LongTensor).to(device)
            
            with torch.set_grad_enabled(False):
                with torch.cuda.amp.autocast():

                    outputs = model(inputs)
                    
                    if config["auto_encoder"]:
                        # Rescale input image to compare it with output image
                        first_channel = inputs[:,:1,:,:]
                        input_scaled = F.interpolate(first_channel, size=(32, 32), mode='bilinear', align_corners=False)
                        loss = loss_function(outputs, input_scaled)
                        
                        if True: # TODO
                            # Extract the slice (single channel image) from the tensor
                            image_slice_in = input_scaled[0, 0, :, :]
                            image_slice_out = outputs[0, 0, :, :]

                            images_max = max(image_slice_in.max(), image_slice_out.max())
                            images_min = min(image_slice_in.min(), image_slice_out.min())
                            image_in = (255 * (image_slice_in - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                            image_out = (255 * (image_slice_out - images_min) / (images_max - images_min)).clamp(0, 255).byte()

                            to_pil = ToPILImage()
                            image_in_pil = to_pil(image_in)
                            image_out_pil = to_pil(image_out)

                            # Save the image as a PNG file
                            img_dir = save_path_cv / "example_images/"
                            os.makedirs(img_dir, exist_ok = True)
                            image_in_pil.save(img_dir / f"{i}_input.png", "PNG")
                            image_out_pil.save(img_dir / f"{i}_output.png", "PNG")

                    else:
                        loss = loss_function(outputs, labels)
            
            if i == 0:
                loss_all_tensor = loss.unsqueeze(0)
                if not config["auto_encoder"]:
                    predictions_tensor = outputs
                    targets_tensor = labels
            else:
                loss_all_tensor = torch.cat([loss_all_tensor, loss.unsqueeze(0)], 0)  # input (t*batch, cxbxwxh)
                if not config["auto_encoder"]:
                    predictions_tensor = torch.cat([predictions_tensor, outputs], 0)
                    targets_tensor = torch.cat([targets_tensor, labels], 0)
        
        # To numpy arrays
        self.mean_loss = np.mean(loss_all_tensor.cpu().numpy())
        if not config["auto_encoder"]:
            self.predictions = predictions_tensor.cpu().numpy()
            self.targets = targets_tensor.cpu().numpy()
            
            self.metrics = self.calc_metrics(self.predictions, self.targets, targets_tensor, config['num_out'])
    
    def calc_metrics(self, predictions, targets, targets_tensor, num_out):
        accuracy = np.mean(np.equal(np.argmax(predictions, 1), targets))
        conf_matrix = confusion_matrix(targets, np.argmax(predictions, 1))
        print("Confusion matrix:\n", conf_matrix)
        weighted_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        sensitivity = np.zeros([num_out])
        specificity = np.zeros([num_out])
        if num_out > 2:
            # will currently result in error if not all classes are in test set...
            if conf_matrix.shape[0] == num_out:
                for k in range(num_out):
                    sensitivity[k] = conf_matrix[k, k] / (np.sum(conf_matrix[k, :]))
                    true_negative = np.delete(conf_matrix, [k], 0)
                    true_negative = np.delete(true_negative, [k], 1)
                    true_negative = np.sum(true_negative)
                    false_positive = np.delete(conf_matrix, [k], 0)
                    false_positive = np.sum(false_positive[:, k])
                    specificity[k] = true_negative / (true_negative + false_positive)
            else:
                tn, fp, fn, tp = conf_matrix.ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
            # F1 score
            f1 = f1_score(targets, np.argmax(predictions, 1), average='weighted')
        else:
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            # F1 score
            f1 = f1_score(targets, np.argmax(predictions, 1))
        # Balanced accuracy
        bacc = (sensitivity + specificity) / 2
        prec = tp / (tp + fp)
        # Matthews Correlation Coefficient
        if (tp+fp) != 0 and (tp+fn) != 0 and (tn+fp) != 0 and (tn+fn) != 0:
            mcc = ((tp * tn) - (fp * fn)) / math.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )
        else:
            mcc = 0.0
        # AUC
        fpr = {}
        tpr = {}
        roc_auc = np.zeros([num_out])
        targets_one_hot = np.array(torch.nn.functional.one_hot(targets_tensor, num_out).cpu().numpy())
        for i in range(num_out):
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return [accuracy, sensitivity, specificity, conf_matrix, f1, roc_auc, weighted_accuracy, bacc, mcc, prec]
    
    