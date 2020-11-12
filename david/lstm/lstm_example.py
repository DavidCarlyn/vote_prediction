import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import pickle

from progress.bar import Bar

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
        #f_t
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
        #c_t
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
        #o_t
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
        
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

def get_dataloader():
    label_file = "outcome_summary.csv"
    feature_file = "D:\\Datasets\\VotePrediction\\combined\\features.pkl"
    metadata_file = "D:\\Datasets\\VotePrediction\\combined\\metadata.pkl"

    lbl_map = {}
    with open(label_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        row = next(csv_reader, None)
        while row is not None:
            try:
                row = next(csv_reader, None)
                case = row[3]
                lbl = int(row[8])
                lbl_map[case] = lbl
            except:
                print(f"Error: {sys.exc_info()[0]}")

    features = read_pkl(feature_file)
    metadata = read_pkl(metadata_file)
    assert len(features) == len(metadata), "Metadata and feature files should match in length"

    cases = []
    for i, feat in enumerate(features):
        if not metadata[i]['valid']: continue
        case = metadata[i]['case']
        if case not in lbl_map.keys(): continue
        sentences = []
        for sen in feat:
            if sen is None: continue
            sentences.append(sen)
        if len(sentences) <= 0: continue
        label = torch.FloatTensor([lbl_map[case]])
        cases.append((torch.Tensor(sentences).view(-1, 226), label))

    return cases

lstm = NaiveCustomLSTM(226, 1)
lstm.train()
optimizer = optim.SGD(lstm.parameters(), lr=0.01)
loss_function = nn.BCELoss()

dataloader = get_dataloader()
train_data = dataloader[:int(len(dataloader)*.8)]
test_data = dataloader[int(len(dataloader)*.8):]

for epoch in range(10):
    total_loss = 0
    i = 0
    for data, lbl in Bar(f"Epoch: {epoch}").iter(train_data):
        i += 1
        size = data.size()
        hidden, output = lstm(data.view(1, size[0], size[1]))
        loss = loss_function(torch.sigmoid(output[1]), lbl.view(1, 1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print()
            print(loss.item())

    print(total_loss / len(train_data))

lstm.eval()
correct = 0
for data, lbl in Bar(f"Epoch: {epoch}").iter(test_data):
    size = data.size()
    hidden, output = lstm(data.view(1, size[0], size[1]))
    pred = int(round(torch.sigmoid(output[1]).item()))
    if int(lbl.item()) == pred:
        correct += 1

print(f"Accuracy: {correct / len(test_data)}")

