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

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_model(layer_sizes=[5, 5, 5], activations=['relu', 'relu', 'relu'], input_size=226, output_size=1, output_activation='sigmoid', bias=True):
    assert len(layer_sizes) == len(activations)
    parameters = []
    last_size = input_size

    for i, s in enumerate(layer_sizes):
        parameters.append(nn.Linear(last_size, int(s), bias=bias))
        if activations[i] == 'relu':
            parameters.append(nn.ReLU())
        elif activations[i] == 'sigmoid':
            parameters.append(nn.Sigmoid())
        else:
            assert False, "Need activation to be ReLU or Sigmoid"
        
        last_size = int(s)

    parameters.append(nn.Linear(last_size, output_size, bias=bias))

    if output_activation == 'sigmoid':
        parameters.append(nn.Sigmoid())
    elif output_activation == 'relu':
        parameters.append(nn.ReLU())
    else:
        assert False, "Need last activation to be ReLU or Sigmoid"

    model = nn.Sequential(*parameters)
    return model

# For a NN we will average the sentence features together
def prep_data(path):
    data = read_pkl(path)
    prepped_data = {}
    mean, std = 0, 0
    for phase in ["train", "test"]:
        audio = data[f"{phase}_audio_data"]
        text = data[f"{phase}_text_data"]
        combined = np.concatenate((text, audio), 2)
        labels = data[f"{phase}_label"]
        mask = data[f"{phase}_mask"]
        average = []
        filt_labels = []

        # Average our features together and extract the labels
        for i in range(combined.shape[0]):
            m = mask[i]
            com = combined[i]
            non_zeros = com[m == 1]
            average.append(non_zeros.mean(0))
            filt_labels.append(1 if labels[i][0][0] == 0 else 0)

        lbls = np.array(filt_labels).astype(np.float32)
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

def create_dataloader(data, batch_size=1, random=False):
    combined = np.concatenate((data[0], data[1].reshape((-1, 1))), axis=1)
    if random:
        np.random.shuffle(combined)
    batches = []
    for i in range(math.ceil(combined.shape[0] / batch_size)):
        start = i * batch_size
        end = (i+1) * batch_size
        features = combined[start:end, :-1]
        target = combined[start:end, -1]
        if i == math.ceil(combined.shape[0] / batch_size) - 1:
            features = combined[start:, :-1]
            target = combined[start:, -1]
        target.reshape((-1, 1))
        batches.append((torch.from_numpy(features), torch.from_numpy(target)))

    return iter(batches)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/datas_new_nor.pkl')
    parser.add_argument('--layer_sizes', nargs='*', default=[200, 300, 500, 300, 200])
    parser.add_argument('--layer_activations', nargs='*', default=['relu', 'relu', 'relu', 'relu', 'relu'])
    parser.add_argument('--output_activation', type=str, default='sigmoid')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--stop_thresh', type=float, default=0.05)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--bias', action='store_true', default=False)
    
    return parser.parse_args()

if __name__ == "__main__":
    # Prep Data
    data = prep_data("../data/datas_new_nor.pkl")
    
    # Get user args
    args = get_args()

    # Random Data Test
    #data = np.random.rand(100, 5).astype(np.float32)
    #r_labels = np.random.rand(100).astype(np.float32)
    #r_labels[r_labels >= 0.5] = 1
    #r_labels[r_labels < 0.5] = 0

    #TEST SVM
    #clf = svm.SVC()
    #clf.fit(data["train"][0], data["train"][1].reshape(-1))
    #preds = clf.predict(data["train"][0])
    #print((preds ==data["train"][1]).sum() / preds.shape[0])

    # Prep Model
    model = create_model(input_size=226, output_size=1, layer_sizes=args.layer_sizes, activations=args.layer_activations, output_activation=args.output_activation, bias=args.bias)
    model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_function = nn.BCELoss()
    if args.loss == 'mse':
        loss_function = nn.MSELoss()

    # For plotting
    plot_points = []
    best_auc = 0
    best_fpr, best_tpr = 0, 0
    best_epoch = 0

    # Train Loop
    for epoch in range(args.epochs):
        model.train()
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

        point_loss = total_loss / num_batches
        acc = correct / total
        print(f"{epoch+1}| Training - Loss: {point_loss}, Accuracy: {acc}")

        # Testing
        test_dataloader = create_dataloader(data["test"], batch_size=args.batch_size, random=False)
        model.eval()

        correct = 0
        total = 0

        raw_output = []
        all_labels = []

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

            # Record Stats
            total += pred.shape[0]
            correct += (pred.reshape(-1) == labels.cpu().detach().numpy()).sum()

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
        if auc_score > best_auc:
            best_auc = auc_score
            best_tpr, best_fpr = tpr, fpr
            best_epoch = epoch
        
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