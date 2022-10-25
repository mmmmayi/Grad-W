import torch
import torch.nn as nn
import torch.nn.functional as F
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))
        self.flipped_filter = torch.load('flipped_filter.pt')

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class BLSTM(nn.Module):
    def __init__(self, configs):
        super(BLSTM, self).__init__()
        self.input_size = configs["input_dim"]
        self.hidden_size = configs["hidden_units"]
        self.num_layers = configs["num_layers"]
        self.output_dim = configs["output_dim"]

        self.stack_blstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout = 0.4)
        self.fconn = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.output_dim)
        self.fconn2 = nn.Linear(
            in_features=self.output_dim,
            out_features=self.output_dim)
        self.activation = nn.Sigmoid()
    def forward(self, X, x_frame):
        X = torch.cat((X,x_frame),dim=1) 
        X = X.permute(0,2,1)
        h0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size).float().cuda()
        c0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size).float().cuda()

        self.stack_blstm.flatten_parameters()
        o, h = self.stack_blstm(X, (h0, c0))
        o = self.fconn(o)
        y_ = self.activation(o)
        return y_.permute(0,2,1)
