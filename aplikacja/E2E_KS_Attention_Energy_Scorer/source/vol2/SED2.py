import torch
import torch.nn as nn
import torch.functional as F


MAX_LENGTH = 256

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        if not hidden:
            output, hidden = self.gru(output)
        else:
            output, hidden = self.gru(output, hidden)
        return output, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden = None):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        if not hidden:
            output, hidden = self.gru(output)
        else:
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, self.device)


# TODO: pamietac ze device jest wczesniej
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.fc_hidden = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)
        self.alignment_vector = nn.Parameter(torch.Tensor(1, hidden_size))
        torch.nn.init.xavier_uniform_(self.alignment_vector)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1)
        embedded = self.dropout(embedded)

        transformed_hidden = self.fc_hidden(hidden[0])
        # bird -? bbbbbbiiiiiiiirrrrrrrdddd
        # TODO: co z MAX_LENGTH (zastapic czyms innym?)
        expanded_hidden_state = transformed_hidden.expand(self.max_length, -1)
        alignment_scores = torch.tanh(expanded_hidden_state +
                                      self.fc_encoder(encoder_outputs))
        alignment_scores = self.alignment_vector.mm(alignment_scores.T)
        attn_weights = F.softmax(alignment_scores, dim=1)
        context_vector = attn_weights.mm(encoder_outputs)

        output = torch.cat((embedded, context_vector), 1).unsqueeze(0)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, self.device)
