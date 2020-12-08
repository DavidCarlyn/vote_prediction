#######################################################################
# Written by: David Carlyn
#######################################################################

import os
import math
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from argparse import ArgumentParser

#######################################################################
# Input:
#     path: the file path containing the pickle file
# Output:
#     a loaded pickle data structure
#######################################################################
def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

#######################################################################
# Input:
#     layer_sizes: The number of units for each hidden layer into the network
#     activations: The activation for each hidden layer of the network
#     input_size: The number of features being passed into the network
#     output_size: The output size of the network
#     output_activation: The output activation of the network
#     bias: True if bias units are included in the neural network
# Output:
#     a Sequential model that is reflective of the input parameters
#######################################################################
def create_model(layer_sizes=[5, 5, 5], activations=['relu', 'relu', 'relu'], input_size=226, output_size=1, output_activation='sigmoid', bias=True):
    # Assertion & starting variables
    assert len(layer_sizes) == len(activations)
    parameters = []
    last_size = input_size

    # Iteratively build each layer of the network
    for i, s in enumerate(layer_sizes):
        parameters.append(nn.Linear(last_size, int(s), bias=bias))
        if activations[i] == 'relu':
            parameters.append(nn.ReLU())
        elif activations[i] == 'sigmoid':
            parameters.append(nn.Sigmoid())
        else:
            assert False, "Need activation to be ReLU or Sigmoid"
        
        last_size = int(s)

    # Add on the last weights of the neural network to fit the output size.
    parameters.append(nn.Linear(last_size, output_size, bias=bias))

    # Add on the output activation
    if output_activation == 'sigmoid':
        parameters.append(nn.Sigmoid())
    elif output_activation == 'relu':
        parameters.append(nn.ReLU())
    else:
        assert False, "Need last activation to be ReLU or Sigmoid"

    # Create and return the model
    model = nn.Sequential(*parameters)
    return model

#######################################################################
# Input:
#     path: The path to the data used for this project
# Output:
#     a data structure that will work for this experiment
#######################################################################
def prep_data(path):
    # Read in the data
    data = read_pkl(path)
    
    prepped_data = {}
    mean, std = 0, 0

    # Prep data per phase (train & test)
    for phase in ["train", "test"]:
        # Extract our desired data
        audio = data[f"{phase}_audio_data"]
        text = data[f"{phase}_text_data"]
        combined = np.concatenate((text, audio), 2)
        labels = data[f"{phase}_label"]
        mask = data[f"{phase}_mask"]

        average = []
        filt_labels = []

        # Average our features together and extract the labels
        for i in range(combined.shape[0]):
            # We want to remove zero rows before we average sequential data
            # Note: there are zero rows for the sequential model to make an equal size
            m = mask[i]
            com = combined[i]
            non_zeros = com[m == 1]

            # Average our values and save them
            average.append(non_zeros.mean(0))
            filt_labels.append(1 if labels[i][0][0] == 0 else 0)

        # Save our labels
        lbls = np.array(filt_labels).astype(np.float32)
        
        # Save our filtered data
        mean_data = np.array(average).astype(np.float32)
        prepped_data[phase] = [mean_data, lbls]

    # Calc mean and std of entire averaged dataset
    all_data = np.concatenate((prepped_data["train"][0], prepped_data["test"][0]), axis=0)
    mean = all_data.mean(0)
    std = all_data.std(0)

    # Normalize data
    prepped_data["train"][0] = (prepped_data["train"][0] - mean) / std
    prepped_data["test"][0] = (prepped_data["test"][0] - mean) / std

    return prepped_data

#######################################################################
# Input:
#     data: the training or testing data
#     batch_size: the size per batch
#     random: True if the data should be shuffled
# Output:
#     an iterable of our data for training / testing our model
#######################################################################
def create_dataloader(data, batch_size=1, random=False):
    # Combine our features and labels before shuffling
    combined = np.concatenate((data[0], data[1].reshape((-1, 1))), axis=1)
    
    # Shuffle our data
    if random:
        np.random.shuffle(combined)
    
    batches = []
    # Create our batches
    for i in range(math.ceil(combined.shape[0] / batch_size)):
        # Indexing
        start = i * batch_size
        end = (i+1) * batch_size

        # Extracting
        features = combined[start:end, :-1]
        target = combined[start:end, -1]

        # For the final batch
        if i == math.ceil(combined.shape[0] / batch_size) - 1:
            features = combined[start:, :-1]
            target = combined[start:, -1]

        # Reshape and save
        target.reshape((-1, 1))
        batches.append((torch.from_numpy(features), torch.from_numpy(target)))

    return iter(batches)

#######################################################################
# Input:
#     NONE
# Output:
#     an object with all the user defined command line arguments
#######################################################################
def get_args():
    parser = ArgumentParser()
    # Path to our data
    parser.add_argument('--data_path', type=str, default='datas_new_nor.pkl')
    # Number of units per hidden layer
    parser.add_argument('--layer_sizes', nargs='*', default=[300, 300, 300])
    # Activation per hidden layer
    parser.add_argument('--layer_activations', nargs='*', default=['relu', 'relu', 'relu'])
    # Output activation
    parser.add_argument('--output_activation', type=str, default='sigmoid')
    # Number of epochs to train for
    parser.add_argument('--epochs', type=int, default=50)
    # The minimum loss values before quitting the training process
    parser.add_argument('--stop_thresh', type=float, default=0.00)
    # The type of loss (bce or mse)
    parser.add_argument('--loss', type=str, default='bce')
    # The type of optimizer to use (adam or SGD)
    parser.add_argument('--optim', type=str, default="adam")
    # The learning rate for training the model
    parser.add_argument('--lr', type=float, default=0.003)
    # The moment for training the model
    parser.add_argument('--momentum', type=float, default=0.9)
    # The size per batch
    parser.add_argument('--batch_size', type=int, default=5)
    # Whether to shuffle the data every epoch
    parser.add_argument('--shuffle', action='store_true', default=False)
    # Whether to include bias terms in the model
    parser.add_argument('--bias', action='store_true', default=False)
    # The type of experiment to run
    parser.add_argument('--exp', type=str, default="both", choices=["both", "text", "audio"])
    
    return parser.parse_args()


#######################################################################
# MAIN
#######################################################################
if __name__ == "__main__":
    # Get user args
    args = get_args()

    # Prep Data
    data = prep_data("../data/datas_new_nor.pkl")
    in_size = 226

    # Filter features if specified
    if args.exp == "text":
        in_size = 200
        data["train"][0] = data["train"][0][:, :200]
        data["test"][0] = data["test"][0][:, :200]
    elif args.exp == "audio":
        in_size = 26
        data["train"][0] = data["train"][0][:, 200:]
        data["test"][0] = data["test"][0][:, 200:]

    # Prep Model
    model = create_model(input_size=in_size, output_size=1, layer_sizes=args.layer_sizes, activations=args.layer_activations, output_activation=args.output_activation, bias=args.bias)
    model.cuda()
    model.train()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss Function
    loss_function = nn.BCELoss()
    if args.loss == 'mse':
        loss_function = nn.MSELoss()

    # For plotting
    plot_points = []
    best_auc = 0
    best_fpr, best_tpr = 0, 0
    best_epoch = 0
    best_acc = 0
    best_matrix = []

    # Train Loop
    for epoch in range(args.epochs):
        model.train()

        # Create dataloader
        train_dataloader = create_dataloader(data["train"], batch_size=args.batch_size, random=args.shuffle)

        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        # Per Batch
        for features, labels in train_dataloader:
            num_batches += 1

            # Forward Pass
            out = model(features.cuda())

            # Capture Predicitons
            pred = np.ones_like(out.cpu().detach().numpy()).astype(np.float32)
            pred[out.cpu().detach().numpy() < 0.5] = 0

            # Calc Loss
            loss = loss_function(out, labels.view(-1, 1).cuda())
            total_loss += loss.item()

            # Record Stats
            total += pred.shape[0]
            cor = (pred.reshape(-1) == labels.cpu().detach().numpy()).sum()
            correct += cor

            # Backward Prop & Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()

        # calc stats
        point_loss = total_loss / num_batches
        acc = correct / total
        print(f"{epoch+1}| Training - Loss: {point_loss}, Accuracy: {acc}")

        # Testing
        test_dataloader = create_dataloader(data["test"], batch_size=args.batch_size, random=False)
        model.eval()

        # Stat variables
        correct = 0
        total = 0

        raw_output = []
        all_labels = []
        predictions = []

        # Per Batch
        for features, labels in test_dataloader:
        
            # Forward Pass
            out = model(features.cuda())

            # Record Raw output & labels
            raw_output.extend(out.cpu().detach().numpy().tolist())
            all_labels.extend(labels.cpu().detach().numpy().reshape(-1).tolist())

            # Capture Predicitons
            pred = np.ones_like(out.cpu().detach().numpy()).astype(np.float32)
            pred[out.cpu().detach().numpy() < 0.5] = 0
            predictions.extend(pred.reshape(-1).tolist())

            # Record Stats
            total += pred.shape[0]
            correct += (pred.reshape(-1) == labels.cpu().detach().numpy()).sum()

        # Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(all_labels, predictions)

        # Plotting AUC curve
        raw_output = np.array(raw_output).reshape(-1)
        all_labels = np.array(all_labels).reshape(-1)
    
        # Get AUC stats
        fpr, tpr, _ = metrics.roc_curve(all_labels, raw_output, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)

        print(f"{epoch+1}| Testing - AUC: {auc_score}, Accuracy: {correct / total}")

        # Record plot points
        plot_points.append([point_loss, acc, correct / total, auc_score])

        # Keep best AUC points
        if auc_score > best_auc and epoch > 9:
            best_auc = auc_score
            best_tpr, best_fpr = tpr, fpr
            best_epoch = epoch
            best_acc = correct / total
            best_matrix = confusion_matrix
        
        # Stop early if loss is less than threshold
        if point_loss < args.stop_thresh:
            break

    # Plot Loss curve
    plot_points = np.array(plot_points)
    plt.plot(np.arange(plot_points.shape[0]), plot_points[:,0])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    #Plot Accuracy curve
    plot_points = np.array(plot_points)
    plt.plot(np.arange(plot_points.shape[0]), plot_points[:,1], c='b', label="Training")
    plt.plot(np.arange(plot_points.shape[0]), plot_points[:,2], c='r', label="Testing")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()

    #Plot AUC scores
    plot_points = np.array(plot_points)
    plt.plot(np.arange(plot_points.shape[0]), plot_points[:,3])
    plt.xlabel('Epochs')
    plt.ylabel('AUC Scores')
    plt.title('AUC Score')
    plt.show()

    # Plot AUC curve
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC curve: {best_auc} at epoch {best_epoch}')
    plt.show()

    # Print Best Results
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Accuracy: {best_acc}")
    print(f"Best AUC: {best_auc}")
    print("Best matrix")
    print(best_matrix)