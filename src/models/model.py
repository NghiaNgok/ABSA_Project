import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class AttentionLSTM(nn.Module):
    def __init__(self):
        super(AttentionLSTM, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(256, 1)
        self.fc = nn.Linear(256, 3)

    def forward(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        bert_out = self.bert(**tokens)[0]
        lstm_out, _ = self.lstm(bert_out)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = attn_weights * lstm_out
        context_vector = torch.sum(context_vector, dim=1)
        output = self.fc(context_vector)
        return output
