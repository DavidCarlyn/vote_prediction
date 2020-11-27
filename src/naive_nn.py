import os
import math
import pickle
import torch
import numpy as np

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_model(layer_sizes=[5, 5, 5], activations=['relu', 'relu', 'relu'], input_size=226, output_size=1, output_activation='sigmoid', bias=True):
    assert len(layer_sizes) == len(activations)
    parameters = []
    last_size = input_size

    for i, s in enumerate(layer_sizes):
        parameters.append(nn.Linear(last_size, s, bias=bias))
        if activations[i] == 'relu':
            parameters.append(nn.ReLU())
        elif activations[i] == 'sigmoid':
            parameters.append(nn.Sigmoid())
        else:
            assert False, "Need activation to be ReLU or Sigmoid"
        
        last_size = s

    parameters.append(nn.Linear(last_size, output_size, bias=bias))

    if output_activation == 'sigmoid':
        parameters.append(nn.Sigmoid())
    elif output_activation == 'relu':
        parameters.append(nn.ReLU())
    else:
        assert False, "Need last activation to be ReLU or Sigmoid"

    model = nn.Sequential(*parameters)
    return model

def prep_data(path):
    data = read_pkl(path)
    prepped_data = {}
    for phase in ["train", "test"]:
        audio = data[f"{phase}_audio_data"]
        text = data[f"{phase}_text_data"]
        combined = np.concatenate((text, audio), 2)
        labels = data[f"{phase}_label"]
        mask = data[f"{phase}_mask"]
        average = []
        filt_labels = []
        for i in range(combined.shape[0]):
            m = mask[i]
            com = combined[i]
            non_zeros = com[m == 1]
            average.append(non_zeros.mean(0))
            filt_labels.append(1 if labels[i][0][0] == 0 else 0)
        prepped_data[phase] = (np.array(average).astype(np.float32), np.array(filt_labels).astype(np.float32))

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
    parser.add_argument('--layer_sizes', nargs='*', default=[2000, 2000, 2000, 2000, 2000, 2000])
    parser.add_argument('--layer_activations', nargs='*', default=['relu', 'relu', 'relu', 'relu', 'relu', 'relu'])
    parser.add_argument('--output_activation', type=str, default='sigmoid')
    parser.add_argument('--epochs', type=int, default=300)
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

    # Train Loop
    for epoch in range(args.epochs):
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

        print(f"{epoch+1}| Loss: {total_loss / num_batches}, Accuracy: {correct / total}")

    # Testing
    test_dataloader = create_dataloader(data["test"], batch_size=args.batch_size, random=False)
    model.eval()

    correct = 0
    total = 0

    # Per Batch
    for features, labels in test_dataloader:
        
        # Forward Pass
        out = model(features.cuda())

        # Capture Predicitons
        pred = np.ones_like(out.cpu().detach().numpy()).astype(np.float32)
        pred[out.cpu().detach().numpy() < 0.5] = 0

        # Record Stats
        total += pred.shape[0]
        correct += (pred.reshape(-1) == labels.cpu().detach().numpy()).sum()

    print(f"Test Accuracy: {correct / total}")