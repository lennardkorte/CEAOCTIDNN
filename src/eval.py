
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from pathlib import Path
import os
from PIL import Image
from torchvision.transforms import ToPILImage
from image_transforms import CircularMask
from scipy.stats import linregress

from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve

class Eval():
    def __init__(self, dataloader, device, model, config, save_path_cv, cv, checkpoint_name=None, class_weights=None):
        model.eval()
        
        n_passes = 1
        if config['MCdropout_test']:
            n_passes = 3
            predictions_list = []

        for _ in range(n_passes):
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels_long = labels.type(torch.LongTensor).to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(False):
                    with torch.cuda.amp.autocast():
                        
                        if config['MCdropout_test']: # TODO
                            for m in model.modules():
                                if m.__class__.__name__.startswith('Dropout'):
                                    m.train()
                                    print("yay")

                            exit()

                        outputs = model(inputs)
                        
                        if config["auto_encoder"]:
                            mse_loss_function = nn.MSELoss(reduction='none')
                            loss_elementwise = mse_loss_function(outputs, inputs)
                            loss_each = torch.mean(loss_elementwise, dim=[1,2,3]) # loss for each image in batch (8 floats for batch size 8)
                            
                            # Compute Loss without log # TODO: mask for training as well
                            me_loss_function = nn.L1Loss(reduction='none') # Mean Error
                            inputs = CircularMask(0.9)(inputs)
                            inputs = CircularMask(0.17, True)(inputs)
                            outputs = CircularMask(0.9)(outputs)
                            outputs = CircularMask(0.17, True)(outputs)
                            loss_linear_elementwise = me_loss_function(outputs, inputs)
                            loss_linear_each = torch.mean(loss_linear_elementwise, dim=[1,2,3])
                        else:
                            ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='none') # no weights here
                            loss_each = ce_loss(outputs, labels_long)
                            
                            # cross-entropy loss without log
                            softmax_function = nn.Softmax(dim=1)
                            outputs_softmax = softmax_function(outputs)
                            nllloss_function = nn.NLLLoss(reduction='none')
                            loss_linear_each = nllloss_function(outputs_softmax, labels_long)
                    
                if i == 0:
                    loss_all_tensor = loss_each
                    loss_linear_all_tensor = loss_linear_each
                    targets_all_tensor = labels_long
                    predictions_all_tensor = outputs
                    inputs_all_tensor = inputs
                        
                else:
                    loss_all_tensor = torch.cat([loss_all_tensor, loss_each], 0)
                    loss_linear_all_tensor = torch.cat([loss_linear_all_tensor, loss_linear_each], 0)
                    targets_all_tensor = torch.cat([targets_all_tensor, labels_long], 0)
                    predictions_all_tensor = torch.cat([predictions_all_tensor, outputs], 0)
                    inputs_all_tensor = torch.cat([inputs_all_tensor, inputs], 0)
                        

            # To numpy arrays
            loss_all = loss_all_tensor.cpu().numpy()
            loss_linear_all = loss_linear_all_tensor.cpu().numpy()
            self.mean_loss = np.mean(loss_all)
            self.mean_loss_linear = np.mean(loss_linear_all)
            targets_np = targets_all_tensor.cpu().numpy()
            predictions_np = predictions_all_tensor.cpu().numpy()

            if config['MCdropout_test']: # TODO
                predictions_list.append(predictions_np)
        
        if config['MCdropout_test']: # TODO
            # Compute entropy for each index

            variances = np.var(predictions_list, axis=0)
            print(variances[0])
            exit()
            '''
            entropies = []
            for index in range(len(predictions_list[0])):
                # Extract the specific index values across all predictions
                index_values = [prediction[index] for prediction in predictions_list]
                # Calculate entropy for this index
                index_entropy = entropy(index_values)
                entropies.append(index_entropy)
            '''

        if not config["auto_encoder"]:
            self.metrics = self.calc_metrics(predictions_np, targets_all_tensor, config['num_classes'])
        
        if checkpoint_name is not None:
            if not config["auto_encoder"]:
                classifier_predictions = np.argmax(predictions_np, 1)

                # Writing integers to the file
                predictions_file_name = save_path_cv / (checkpoint_name + '_predloss_pairs.txt')
                with open(predictions_file_name, 'w') as file:
                    for int_class, float_loss in zip(classifier_predictions,loss_linear_all):
                        file.write(f"{int_class},{float_loss}\n")

            else:
                self.save_auto_encoder_sample(inputs_all_tensor, predictions_all_tensor, save_path_cv)

                if config['compare_classifier_predictions']:

                    # Load classifier_predictions, classifier_losses for checkpoint_name
                    save_path_classifier = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(cv)))
                    predictions_file_name = save_path_classifier / (checkpoint_name + '_predloss_pairs.txt')
                    classifier_predloss_pairs = []
                    with open(predictions_file_name, 'r') as file:
                        for line in file:
                            parts = line.strip().split(',')
                            classifier_predloss_pairs.append((int(parts[0]), float(parts[1])))
                    classifier_predictions, classifier_losses = tuple(map(list, zip(*classifier_predloss_pairs)))
                    
                    self.mse_loss_conf_matr_mean = self.calc_mse_loss_conf_matr_mean(loss_linear_all, targets_np, classifier_predictions)

                    # Apply Threshold
                    classifier_losses = np.array(classifier_losses)
                    mask1 = np.array([elem > -1 for elem in classifier_losses])
                    mask2 = np.array([elem > -10.0 for elem in loss_linear_all])
                    classifier_losses_threshold = classifier_losses[mask1 & mask2]
                    loss_linear_all_threshold = loss_linear_all[mask1 & mask2]

                    # New font sizes for both plots
                    title_fontsize = 20
                    axis_label_fontsize = 20
                    tick_label_fontsize = 15

                    # Create plot Loss Pair Distribution of Classifier vs Autoencoder
                    colors = np.array([])
                    for t, p in zip(targets_np, classifier_predictions): #colors = np.array([('green' if t == p else 'red') for t, p in zip(targets_all_tensor, classifier_predictions)])
                        color = 'green' if t == p else 'red' # TODO: colors not right
                        colors = np.append(colors, color)
                    colors_threshold = colors[mask1 & mask2]
                    plt.clf()
                    plt.figure(figsize=(10, 6))

                    x = loss_linear_all_threshold
                    y = classifier_losses_threshold
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    line = slope * x + intercept
                    pearson_coefficient = r_value
                    print("slope: ", slope)
                    print("pearson_coefficient: ", pearson_coefficient)

                    plt.scatter(loss_linear_all_threshold, classifier_losses_threshold, alpha=0.5, c=colors_threshold)
                    plt.plot(x, line, color='blue', label='Linear Regression Line')
                    #plt.title('Loss Pair Distribution of Classifier vs Autoencoder', fontsize=title_fontsize)
                    plt.xlabel('x', fontsize=axis_label_fontsize)
                    plt.ylabel('y', fontsize=axis_label_fontsize)
                    plt.xticks(fontsize=tick_label_fontsize)  # Larger x-axis tick labels
                    plt.yticks(fontsize=tick_label_fontsize)  # Larger y-axis tick labels
                    plt.grid(True)
                    plt.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.1)
                    plt.savefig(save_path_cv / (checkpoint_name + '_loss_distribution_plot.png'))
                    plt.close()

                    # Create plot with Proportion of Samples with Lowest MAE
                    loss_linear_all_asc, classifier_losses_asc = zip(*sorted(zip(loss_linear_all_threshold, classifier_losses_threshold), key=lambda x: x[0]))
                    loss_linear_all_asc = list(loss_linear_all_asc)
                    classifier_losses_asc = list(classifier_losses_asc)
                    proportions = np.linspace(0.01, 1, 100)  # 100 proportions from 1% to 100%
                    average_losses = [] # average_losses contains the average classifier loss for each proportion
                    for proportion in proportions:
                        n_samples = int(proportion * len(classifier_losses_asc))
                        selected_losses = classifier_losses_asc[:n_samples]
                        average_loss = np.mean(selected_losses)
                        average_losses.append(average_loss)
                    plt.clf()
                    plt.figure(figsize=(10, 6))
                    plt.plot(proportions, average_losses)
                    plt.xlabel('x', fontsize=axis_label_fontsize)
                    plt.ylabel('y', fontsize=axis_label_fontsize)
                    plt.xticks(fontsize=tick_label_fontsize)  # Larger x-axis tick labels
                    plt.yticks(fontsize=tick_label_fontsize)  # Larger y-axis tick labels
                    #plt.title('Relationship between MAE and Classifier Loss', fontsize=title_fontsize)
                    plt.grid(True)
                    plt.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.1)
                    plt.savefig(save_path_cv / (checkpoint_name + '_risk_coverage_curve.png'))
                    plt.close()

    def save_auto_encoder_sample(self, inputs_all_tensor, predictions_all_tensor, save_path_cv):
        num_samples = 10
        for i, (input_tensor, prediction_tensor) in enumerate(zip(inputs_all_tensor, predictions_all_tensor)):
            if not i % int(len(predictions_all_tensor) / num_samples):
                # Extract the slice (single channel image) from the tensor
                image_slice_in = input_tensor[0, :, :]
                image_slice_pred = prediction_tensor[0, :, :]

                '''
                # with filters:
                image_slice_in_filtered = input_tensor[0, :, :]
                image_slice_in_filtered = T.GaussianBlur(kernel_size=(5,5), sigma=(2,2))(image_slice_in_filtered.unsqueeze(0)).squeeze(0)
                from da_techniques import CircularMask
                image_slice_in_filtered = CircularMask(radius=0.1)(image_slice_in_filtered)
'''
                
                images_max = max(image_slice_in.max(), image_slice_pred.max())
                images_min = min(image_slice_in.min(), image_slice_pred.min())
                image_in = (255 * (image_slice_in - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                # image_in_filtered = (255 * (image_slice_in_filtered - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                image_out = (255 * (image_slice_pred - images_min) / (images_max - images_min)).clamp(0, 255).byte()

                to_pil = ToPILImage()
                image_in_pil = to_pil(image_in)
                # image_in_filtered_pil = to_pil(image_in_filtered)
                image_out_pil = to_pil(image_out)
                img_dir = save_path_cv / "example_images/"
                os.makedirs(img_dir, exist_ok = True)
                image_in_pil.save(img_dir / f"{i}_input.png", "PNG")
                # image_in_filtered_pil.save(img_dir / f"{i}_input_blur.png", "PNG")
                image_out_pil.save(img_dir / f"{i}_output.png", "PNG")

                # Print absolute difference of input and output
                abs_diff = torch.abs(torch.subtract(image_slice_in, image_slice_pred))
                abs_diff = (255 * (abs_diff - images_min) / (images_max - images_min)).clamp(0, 255).byte()
                #Alternative min and max values when normalizing for range 0 to 255
                #abs_diff = (255 * (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min())).clamp(0, 255).byte()
                abs_diff_image = abs_diff.cpu().numpy()
                abs_diff_image_np = Image.fromarray(np.uint8(abs_diff_image), mode='L')
                abs_diff_image_np.save(img_dir / f"{i}_absdiff.png")
                '''
                segments = skimage.segmentation.slic(abs_diff_image, n_segments=8, compactness=0.03, channel_axis=None)
                segmentation_overlay = skimage.color.label2rgb(segments, image=abs_diff_image, kind='overlay')
                segmentation_image = Image.fromarray(np.uint8(segmentation_overlay * 255))
                segmentation_image.save(img_dir / f"{i}_segmentation.png")
                '''

    def calc_mse_loss_conf_matr_mean(self, loss_linear_all, classifier_targets, classifier_predictions):
        mse_loss_conf_matr = [[[],[]], [[],[]]]
        for loss_linear, classifier_target, classifier_prediction in zip(loss_linear_all, classifier_targets, classifier_predictions):
            mse_loss_conf_matr[int(classifier_prediction)][int(classifier_target)].append(loss_linear)

        # Calculating the mean of each list and storing it in a 2D matrix
        mse_loss_conf_matr_mean = np.array([[np.mean(lst) for lst in row] for row in mse_loss_conf_matr])

        return mse_loss_conf_matr_mean
    
    def calc_metrics(self, predictions_np, targets, num_classes):
        
        targets_np = targets.cpu().numpy()

        accuracy = np.mean(np.equal(np.argmax(predictions_np, 1), targets_np))
        conf_matrix = confusion_matrix(targets_np, np.argmax(predictions_np, 1))

        weighted_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        sensitivity = np.zeros([num_classes])
        specificity = np.zeros([num_classes])
        if num_classes > 2:
            # will currently result in error if not all classes are in test set...
            if conf_matrix.shape[0] == num_classes:
                for k in range(num_classes):
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
        roc_auc = np.zeros([num_classes])
        targets_one_hot = np.array(torch.nn.functional.one_hot(targets, num_classes).cpu().numpy())
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], predictions_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        self.confusion_matrix = conf_matrix # TODO: unused
        self.roc_auc = roc_auc # TODO: unused
        self.weighted_accuracy = weighted_accuracy # TODO: unused


        return [accuracy, sensitivity, specificity, f1, bacc, mcc, prec]
    
    