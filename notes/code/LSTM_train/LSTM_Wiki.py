from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import time
from sklearn.metrics import top_k_accuracy_score
import numpy as np


class LSTM_B_Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        tie_weights,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            assert embedding_dim == hidden_dim, "cannot tie, check dims"
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(
                self.embedding_dim, self.hidden_dim
            ).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(
                self.hidden_dim, self.hidden_dim
            ).uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell


class CustomTrainer:
    def __init__(self):
        pass

    def train(
        self,
        model,
        train_data,
        epochs,
        device=torch.device("cuda"),
        args: dict = None,
    ):
        def get_batch(data, seq_len, idx):
            src = data[:, idx : idx + seq_len]
            target = data[:, idx + 1 : idx + seq_len + 1]
            return src, target

        def train_one_epoch(
            model, data, optimizer, criterion, batch_size, seq_len, clip, device
        ):
            epoch_loss = 0
            model.to(device)
            model.train()
            num_batches = data.shape[-1]
            data = data[:, : num_batches - (num_batches - 1) % seq_len]
            num_batches = data.shape[-1]

            hidden = model.init_hidden(batch_size, device)
            correct_predictions = 0
            total_num_samples = 0
            targets = list()

            for idx in tqdm(range(0, num_batches - 1, seq_len), desc="Mini-batch", leave=False):
                optimizer.zero_grad()
                hidden = model.detach_hidden(hidden)

                src, target = get_batch(data, seq_len, idx)
                src, target = src.to(device), target.to(device)
                batch_size = src.shape[0]
                prediction, hidden = model(src, hidden)

                prediction = prediction.reshape(batch_size * seq_len, -1)
                target = target.reshape(-1)
                loss = criterion(prediction, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss += loss.item() * seq_len

                correct_predictions += sum(
                    torch.argmax(prediction, dim=1) == target
                ).item()
                total_num_samples += len(target)

            return {
                "loss": epoch_loss / num_batches,
                "accuracy": (correct_predictions / total_num_samples) * 100,
            }

        start_time = time.time()
        seq_len = args["seq_length"]
        clip = args["clip"]
        lr = args["lr"]
        batch_size = args["batch_size"]

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The model has {num_params:,} trainable parameters")
        loss = list()
        accuracy = list()
        for e in tqdm(range(epochs), desc="Training", leave=False):
            train_results = train_one_epoch(
                model=model,
                data=train_data,
                optimizer=optimizer,
                criterion=criterion,
                batch_size=batch_size,
                seq_len=seq_len,
                clip=clip,
                device=device,
            )
            torch.save(model.state_dict(), './model_Jan10.pth')
            loss.append(train_results["loss"])
            accuracy.append(train_results["accuracy"])
            print("Training round ", e, train_results)

        results = {
            "epochs": epochs,
            "loss": sum(loss) / len(loss),
            "accuracy": (sum(accuracy) / len(accuracy)),
            "time_taken_s": time.time() - start_time,
        }
        return results

    def validate_model(
        self,
        model,
        dataloader,
        device=torch.device("cuda"),
        args=None,
    ):
        def get_batch(data, seq_len, num_batches, idx):
            src = data[:, idx : idx + seq_len]
            target = data[:, idx + 1 : idx + seq_len + 1]
            return src, target

        seq_len = args["seq_length"]
        batch_size = args["batch_size"]
        epoch_loss = 0
        model.eval()
        num_batches = dataloader.shape[-1]
        data = dataloader[:, : num_batches - (num_batches - 1) % seq_len]
        num_batches = data.shape[-1]
        criterion = nn.CrossEntropyLoss()

        hidden = model.init_hidden(batch_size, device)
        correct_predictions = 0
        total_num_samples = 0

        with torch.no_grad():
            for idx in tqdm(range(0, num_batches - 1, seq_len), desc="Mini-batch", leave=False):
                hidden = model.detach_hidden(hidden)
                src, target = get_batch(dataloader, seq_len, num_batches, idx)
                src, target = src.to(device), target.to(device)
                batch_size = src.shape[0]

                prediction, hidden = model(src, hidden)
                prediction = prediction.reshape(batch_size * seq_len, -1)
                target = target.reshape(-1)


                loss = criterion(prediction, target)
                epoch_loss += loss.item() * seq_len
                prediction = np.array(prediction.tolist())
                target = np.array(target.tolist())
                correct_predictions = top_k_accuracy_score(y_true=target, y_score=prediction, k=5, normalize=False, labels=np.arange(vocab_size))

                total_num_samples += len(target)

        return {
            "loss": epoch_loss / num_batches,
            "accuracy": correct_predictions / total_num_samples,
        }


class CustomDataLoader:
    def __init__(self) -> None:
        pass

    def get_train_test_dataset_loaders(
        self, batch_size=1, dataset_path=None, args: dict = None
    ):
        train_data = torch.load(
                "WIKITEXT-29423-16/partition_0/train_partition_0.pth"
        )
        
        return train_data#, test_data


if __name__ == "__main__":
    train_data = CustomDataLoader().get_train_test_dataset_loaders(
        args={"num_samples": 798}
    )

    vocab_size = 29423
    embedding_dim = 256 # 400 in the paper
    hidden_dim = 256  # 1150 in the paper
    num_layers = 2  # 3 in the paper
    dropout_rate = 0.2
    tie_weights = True
    seq_length = 8
    batch_size = 16
    flops = 2 * num_layers * seq_length * batch_size * (
    hidden_dim**2 + hidden_dim * embedding_dim + embedding_dim * vocab_size)

    print(f"Forward FLOPs: {flops:,}")

    model = LSTM_B_Model(
        vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights
    )
    print('previosuly saved model loaded')


    print(model)
    custom_trainer = CustomTrainer()
    print(
        custom_trainer.train(
            model,
            train_data,
            epochs=100,
            args={
                "batch_size": batch_size,
                "seq_length": seq_length,
                "clip": 0.5,
                "lr": 5e-5,
            },
        )
    )
    print(
        custom_trainer.validate_model(
            model=model,
            dataloader=train_data,
            args={"seq_length": seq_length, "batch_size": batch_size},
        ),
        None,
    )
    
