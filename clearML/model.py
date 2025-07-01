import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1)  # hidden_size * 2 for BiLSTM

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size * 2)
        scores = self.attention_weights(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = torch.softmax(scores, dim=1)  # Softmax over the sequence length
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # Weighted sum of the outputs
        return context_vector, attention_weights.squeeze(-1)

class ActionRecognitionBiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(ActionRecognitionBiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=self.bidirectional)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_length, hidden_size * 2)
        out = self.dropout(out)

        context_vector, attention_weights = self.attention(out)

        # Decode the context vector from the attention layer
        out = self.fc(context_vector)
        return out, attention_weights
