import torch
import torch.nn as nn


class MyRNN(nn.Module):
    '''rnn(150) -> rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)'''

    def __init__(self, embedded_matrix, rnn='vanilla', h=150, num_layers=2, dropout=0, bidirectional=False,
                 activation_fn=nn.functional.relu):
        super(MyRNN, self).__init__()

        self.embedded_matrix = embedded_matrix
        self.activation_fn = activation_fn
        if rnn == 'vanilla':
            self.rnn = nn.RNN(input_size=300, hidden_size=h, num_layers=num_layers, dropout=0, bidirectional=bidirectional)
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=300, hidden_size=h, num_layers=num_layers, dropout=0, bidirectional=bidirectional)
        elif rnn == 'gru':
            self.rnn = nn.GRU(input_size=300, hidden_size=h, num_layers=num_layers, dropout=0, bidirectional=bidirectional)

        input_size = 2 * h if bidirectional else h
        self.fc1 = nn.Linear(input_size, 150)
        self.fc_logits = nn.Linear(150, 1)

    def forward(self, x):
        x = self.embedded_matrix(x)
        x = torch.transpose(x, 1, 0)  # time-first format
        x, _ = self.rnn(x)
        x = self.fc1(x[-1])
        x = self.activation_fn(x)
        x = self.fc_logits(x)
        return x
