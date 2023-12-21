
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import torch.nn as nn
from pathlib import Path
import cv2 as cv
import os
from PIL import Image
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
import skimage.segmentation

from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve

class Eval():
    def __init__(self, dataloader, device, model, loss_function, config, save_path_cv, cv, if_test=False):
        model.eval()
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels_long = labels.type(torch.LongTensor).to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                with torch.cuda.amp.autocast():

                    outputs = model(inputs)
                    
                    if config["auto_encoder"]:
                        loss_each = loss_function(outputs, inputs)
                        
                        mse_loss_function = nn.L1Loss(reduction='none')
                        loss_mse_each = mse_loss_function(outputs, inputs)
                    else:
                        loss_each = loss_function(outputs, labels_long)
                        
                        softmax_function = nn.Softmax(dim=1)
                        outputs_softmax = softmax_function(outputs)
                        nllloss_function = nn.NLLLoss(reduction='none')
                        loss_mse_each = nllloss_function(outputs_softmax, labels_long)
            
            if config["auto_encoder"]:
                loss_each = torch.mean(loss_each, dim=[1,2,3]) # loss for each image in batch (8 floats for batch size 8)
            
            if i == 0:
                loss_all_tensor = loss_each
                loss_mse_all_tensor = loss_mse_each
                targets_tensor = labels_long
                predictions_tensor = outputs
                inputs_tensor = inputs
                    
            else:
                print(loss_all_tensor, loss_each)
                print(loss_mse_all_tensor, loss_mse_each)
                exit()
                loss_all_tensor = torch.cat([loss_all_tensor, loss_each], 0)
                loss_mse_all_tensor = torch.cat([loss_mse_all_tensor, loss_mse_each], 0)
                targets_tensor = torch.cat([targets_tensor, labels_long], 0)
                predictions_tensor = torch.cat([predictions_tensor, outputs], 0)
                inputs_tensor = torch.cat([inputs_tensor, inputs], 0)
                    

        # To numpy arrays
        loss_all = loss_all_tensor.cpu().numpy()
        loss_mse_all = loss_mse_all_tensor.cpu().numpy()
        self.mean_loss = np.mean(loss_all)
        self.mean_loss_mse = np.mean(loss_mse_all)

        if not config["auto_encoder"]:
            self.metrics = self.calc_metrics(predictions_tensor, targets_tensor, config['num_out'])
            
        
        if if_test:
            if not config["auto_encoder"]:
                predictions_np = predictions_tensor.cpu().numpy()
                classifier_predictions = np.argmax(predictions_np, 1)

                # Writing integers to the file
                predictions_file_name = save_path_cv / 'test_predloss_pairs.txt'
                with open(predictions_file_name, 'w') as file:
                    for int_class, float_loss in zip(classifier_predictions,loss_mse_all):
                        file.write(f"{int_class},{float_loss}\n")

            else:
                self.save_auto_encoder_sample(inputs_tensor, predictions_tensor, save_path_cv)

                if config['compare_classifier_predictions']:

                    # Reading integers from the file and writing them back to another variable
                    # Load the pretrained ResNet18 model from a ".pt" file
                    save_path_classifier = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(cv)))
                    
                    predictions_file_name = save_path_classifier / 'test_predloss_pairs.txt'
                    classifier_predloss_pairs = []
                    with open(predictions_file_name, 'r') as file:
                        for line in file:
                            parts = line.strip().split(',')
                            classifier_predloss_pairs.append((int(parts[0]), float(parts[1])))
                    
                    # print(classifier_predloss_pairs)
                    # exit()
                    classifier_predictions, classifier_losses = tuple(map(list, zip(*classifier_predloss_pairs)))

                    self.mse_loss_conf_matr_mean = self.calc_mse_loss_conf_matr_mean(inputs_tensor, predictions_tensor, targets_tensor, classifier_predictions)

                    # Apply Threshold
                    classifier_losses = np.array(classifier_losses)
                    loss_all = np.array(loss_all)
                    mask1 = np.array([elem < 2.0 for elem in classifier_losses])
                    mask2 = np.array([elem < 0.001 for elem in loss_all])
                    classifier_losses_threshold = classifier_losses[mask1 & mask2]
                    loss_all_threshold = loss_all[mask1 & mask2]

                    # Creating and storing the scatter plot for the larger dataset
                    plot_file_path = save_path_cv / 'loss_distribution_plot.png'
                    colors = []
                    for t, p in zip(targets_tensor, classifier_predictions):
                        color = 'green' if t == p else 'red'
                        colors.append(color)
                    colors = np.array([('green' if t == p else 'red') for t, p in zip(targets_tensor, classifier_predictions)])
                    colors_threshold = colors[mask1 & mask2]
                    plt.clf()
                    plt.figure(figsize=(10, 6))
                    plt.scatter(loss_all_threshold, classifier_losses_threshold, alpha=0.5, c=colors_threshold)
                    plt.title('Loss Pair Distribution of Classifier vs Autoencoder')
                    plt.xlabel('Autoencoder Loss')
                    plt.ylabel('Classifier Loss')
                    plt.grid(True)
                    plt.savefig(plot_file_path)

            
    def save_auto_encoder_sample(self, inputs_tensor, predictions_tensor, save_path_cv):
        num_samples = 20
        for i, (input_tensor, prediction_tensor) in enumerate(zip(inputs_tensor, predictions_tensor)):
            if not i % int(len(predictions_tensor) / num_samples):
                # Extract the slice (single channel image) from the tensor
                image_slice_in = input_tensor[0, :, :]
                image_slice_pred = prediction_tensor[0, :, :]
                # with blurring:
                #image_slice_in_blur = input_scaled[0, :1, :, :].clone().detach()
                #image_slice_in_blur = T.GaussianBlur(kernel_size=(5,5), sigma=(2,2))(image_slice_in_blur)
                #image_slice_in_blur = image_slice_in_blur[0, :, :]
                
                images_max = max(image_slice_in.max(), image_slice_pred.max())
                images_min = min(image_slice_in.min(), image_slice_pred.min())
                image_in = (255 * (image_slice_in - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                #image_in_blur = (255 * (image_slice_in_blur - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                image_out = (255 * (image_slice_pred - images_min) / (images_max - images_min)).clamp(0, 255).byte()

                to_pil = ToPILImage()
                image_in_pil = to_pil(image_in)
                #image_in_blur_pil = to_pil(image_in_blur)
                image_out_pil = to_pil(image_out)

                # Save the image as a PNG file
                img_dir = save_path_cv / "example_images/"
                os.makedirs(img_dir, exist_ok = True)
                image_in_pil.save(img_dir / f"{i}_input.png", "PNG")
                #image_in_blur_pil.save(img_dir / f"{i}_input_blur.png", "PNG")
                image_out_pil.save(img_dir / f"{i}_output.png", "PNG")

                # Print absolute difference of input and output
                abs_diff = torch.abs(torch.subtract(image_slice_in, image_slice_pred))
                abs_diff = (255 * (abs_diff - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                #Alternative min and max values when normalizing for range 0 to 255
                #abs_diff = (255 * (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min())).clamp(0, 255).byte()

                abs_diff_image = abs_diff.cpu().numpy()

                '''
                kernel_size = 5  # Adjust this as needed
                abs_diff_image = cv.medianBlur(abs_diff_image, kernel_size)
                alpha = 2.0  # Contrast control (1.0-3.0)
                beta = 0     # Brightness control (0-100)
                abs_diff_image = cv.convertScaleAbs(abs_diff_image, alpha=alpha, beta=beta)
                #abs_diff_image_con[abs_diff_image_con > 35] = 0
                '''

                # Save the image as a PNG file
                abs_diff_image_np = Image.fromarray(np.uint8(abs_diff_image), mode='L')
                abs_diff_image_np.save(img_dir / f"{i}_absdiff.png")

                segments = skimage.segmentation.slic(abs_diff_image, n_segments=8, compactness=0.03, channel_axis=None)
                segmentation_overlay = skimage.color.label2rgb(segments, image=abs_diff_image, kind='overlay')
                
                # Convert the numpy segmentation to a uint8 image
                segmentation_image = Image.fromarray(np.uint8(segmentation_overlay * 255))

                # Save the image as a PNG file
                segmentation_image.save(img_dir / f"{i}_segmentation.png")

    def calc_mse_loss_conf_matr_mean(self, inputs, predictions, classifier_targets_tensor, classifier_predictions):

        classifier_targets = classifier_targets_tensor.cpu().numpy()

        # Compute MSE Loss for input and output image of autoencoder
        loss_function_mse = nn.MSELoss() # instantiation
        
        mse_loss_conf_matr = [[[],[]], [[],[]]]
        for input, prediction, classifier_target, classifier_prediction in zip(inputs, predictions, classifier_targets, classifier_predictions):
            mse_loss = loss_function_mse(input, prediction).cpu().numpy()
            mse_loss_conf_matr[int(classifier_prediction)][int(classifier_target)].append(mse_loss)

        # Calculating the mean of each list and storing it in a 2D matrix
        mse_loss_conf_matr_mean = np.array([[np.mean(lst) for lst in row] for row in mse_loss_conf_matr])

        return mse_loss_conf_matr_mean
    
    def calc_metrics(self, predictions_tensor, targets, num_out):
        
        targets_np = targets.cpu().numpy()
        predictions_np = predictions_tensor.cpu().numpy()

        accuracy = np.mean(np.equal(np.argmax(predictions_np, 1), targets_np))
        conf_matrix = confusion_matrix(targets_np, np.argmax(predictions_np, 1))

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
            f1 = f1_score(targets_np, np.argmax(predictions_np, 1), average='weighted')
        else:
            tn, fp, fn, tp = conf_matrix.ravel() # TODO: raveling multiple times
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            # F1 score
            f1 = f1_score(targets_np, np.argmax(predictions_np, 1))
        # Balanced accuracy
        bacc = (sensitivity + specificity) / 2
        prec = tp / (tp + fp) if tp + fp != 0 else 0.0 #TODO: check zero div everywhere
        # Matthews Correlation Coefficient
        if (tp+fp) != 0 and (tp+fn) != 0 and (tn+fp) != 0 and (tn+fn) != 0:
            mcc = ((tp * tn) - (fp * fn)) / math.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )
        else:
            mcc = 0.0
        # AUC
        fpr = {}
        tpr = {}
        roc_auc = np.zeros([num_out])
        targets_one_hot = np.array(torch.nn.functional.one_hot(targets, num_out).cpu().numpy())
        for i in range(num_out):
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], predictions_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        self.confusion_matrix = conf_matrix # TODO: unused
        self.roc_auc = roc_auc # TODO: unused
        self.weighted_accuracy = weighted_accuracy # TODO: unused


        return [accuracy, sensitivity, specificity, f1, bacc, mcc, prec]
    
    