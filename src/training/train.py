import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import AttentionLSTM
from datasets.dataset import CustomABSA, collate_fn

def train_model(dataframe, epochs=10, batch_size=8, lr=1e-3):
    custom_dataset = CustomABSA(dataframe)
    custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionLSTM().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sentences, labels in custom_loader:
            sentences = list(sentences)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(sentences)

            if outputs.dim() == 2:
                labels = labels[:, 0].long()
                loss = criterion(outputs, labels)
            else:
                batch_size, seq_len, num_classes = outputs.size()
                outputs = outputs.view(batch_size * seq_len, num_classes)
                labels = labels.view(-1).long()
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(custom_loader):.4f}")

    return model
