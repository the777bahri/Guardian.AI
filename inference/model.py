import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    # ...(keep implementation)
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_output):
        scores = torch.tanh(self.attention_weights(lstm_output))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights.squeeze(-1)

class ActionRecognitionBiLSTMWithAttention(nn.Module):
     # ...(keep implementation)
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(ActionRecognitionBiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        context_vector, attention_weights = self.attention(lstm_out)
        context_vector_dropped = self.dropout(context_vector)
        logits = self.fc(context_vector_dropped)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities, attention_weights