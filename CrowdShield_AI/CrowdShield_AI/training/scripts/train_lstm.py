"""Train the custom LSTM behavior model.

This script provides an end-to-end training workflow with:
- placeholder/mock dataset loading
- train loop
- validation loop
- model checkpoint export to model.pt

The model architecture is imported from custom_lstm.py and uses:
- CrossEntropyLoss
- Adam optimizer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from custom_lstm import CustomLSTMClassifier


@dataclass(frozen=True)
class TrainConfig:
    """Runtime configuration for model training."""

    input_size: int = 10
    seq_len: int = 30
    num_classes: int = 4
    hidden_size: int = 128
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    train_ratio: float = 0.8
    num_samples: int = 2000
    model_output: str = "model.pt"


class MockSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Synthetic placeholder dataset for sequence classification.

    Replace this class with a real loader when ground-truth feature sequences are
    available from the pose/feature pipeline.
    """

    def __init__(self, num_samples: int, seq_len: int, input_size: int, num_classes: int) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_size = input_size
        self.num_classes = num_classes

        generator = torch.Generator().manual_seed(42)

        # Create random labels first, then generate slightly class-dependent
        # sequence statistics so the model can learn during this mock phase.
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        base = torch.randn(num_samples, seq_len, input_size, generator=generator)

        class_offsets = torch.tensor([0.2, 0.6, -0.4, 1.0], dtype=torch.float32)
        offsets = class_offsets[self.labels].view(-1, 1, 1)
        self.features = base + offsets

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.features[index].float()  # (seq_len, input_size)
        y = self.labels[index].long()     # scalar class id
        return x, y


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Build train/validation dataloaders from a placeholder dataset."""
    dataset = MockSequenceDataset(
        num_samples=config.num_samples,
        seq_len=config.seq_len,
        input_size=config.input_size,
        num_classes=config.num_classes,
    )

    train_size = int(len(dataset) * config.train_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(
    model: CustomLSTMClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return (mean_loss, accuracy)."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += batch_size

    mean_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    return mean_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model: CustomLSTMClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (mean_loss, accuracy)."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += batch_size

    mean_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    return mean_loss, accuracy


def save_model(model: CustomLSTMClassifier, output_path: Path) -> None:
    """Save model weights to disk as model.pt (state dict only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train custom LSTM behavior model")
    parser.add_argument("--input-size", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--output", default="model.pt", help="Output model path")

    args = parser.parse_args()
    return TrainConfig(
        input_size=args.input_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
        num_samples=args.num_samples,
        model_output=args.output,
    )


def main() -> None:
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Train] Device: {device}")
    print(f"[Train] Config: {config}")

    train_loader, val_loader = build_dataloaders(config)

    model = CustomLSTMClassifier(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
    ).to(device)

    # Required training components.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"[Epoch {epoch:02d}/{config.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    output_path = Path(config.model_output)
    save_model(model, output_path)
    print(f"[Train] Saved model to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
