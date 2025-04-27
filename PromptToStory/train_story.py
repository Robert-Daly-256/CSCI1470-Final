import os, argparse, pickle
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from preprocess import Vocab
from story_generation_model import TransformerStoryGenerator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='folder with preprocessed .pt files, relative to prompttostory')
    p.add_argument('--vocab',    required=True, help='path to vocab_story.pkl')
    p.add_argument('--epochs',   type=int, default=10)
    p.add_argument('--bs',       type=int, default=16)
    p.add_argument('--lr',       type=float, default=5e-4)
    p.add_argument('--save_dir', default='ckpts')
    return p.parse_args()

def load_split(d, split):
    src = torch.load(f"{d}/X_{split}.pt")
    tgt = torch.load(f"{d}/Y_{split}.pt")
    return TensorDataset(src, tgt)

def train_epoch(m, loader, opt, crit, dev, scaler):
    m.train(); total_loss = 0; total_correct = 0; total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(dev), tgt.to(dev)
        inp, lbl = tgt[:, :-1], tgt[:, 1:]

        opt.zero_grad()

        # Use autocast for mixed precision
        with autocast():
            logits = m(src, inp)  # forward
            loss = crit(logits.view(-1, logits.size(-1)), lbl.reshape(-1))

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()

        # Update the model parameters
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()

        # Calculate accuracy
        pred = logits.argmax(dim=-1)  # Predicted token indices
        correct = (pred == lbl).sum().item()  # Count how many predictions are correct
        total_correct += correct
        total_tokens += lbl.numel()  # Total number of tokens in the batch

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_correct / total_tokens
    return avg_loss, avg_accuracy

def eval_epoch(m, loader, crit, dev):
    m.eval(); total_loss = 0; total_correct = 0; total_tokens = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(dev), tgt.to(dev)
            inp, lbl = tgt[:, :-1], tgt[:, 1:]
            logits = m(src, inp)
            loss = crit(logits.view(-1, logits.size(-1)), lbl.reshape(-1))
            total_loss += loss.item()

            # Calculate accuracy: compare predicted tokens to ground truth
            pred = logits.argmax(dim=-1)  # Predicted token indices
            correct = (pred == lbl).sum().item()  # Count how many predictions are correct
            total_correct += correct
            total_tokens += lbl.numel()  # Total number of tokens in the batch

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_correct / total_tokens
    return avg_loss, avg_accuracy

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_ds = load_split(args.data_dir, 'train')
    val_ds   = load_split(args.data_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.bs)

    # Load vocab for pad index
    vocab = pickle.load(open(args.vocab, 'rb'))
    vocab_size = len(vocab)
    pad_idx    = vocab.padding_index

    # Model, optimizer, loss
    model     = TransformerStoryGenerator(vocab_size).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    scaler = GradScaler()

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_acc = train_epoch(model, train_ld, optimizer, criterion, dev, scaler)
        # Evaluate
        val_loss, val_acc = eval_epoch(model, val_ld, criterion, dev)

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}%  val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        # Save checkpoint
        ckpt = os.path.join(args.save_dir, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

if __name__ == '__main__':
    main()
