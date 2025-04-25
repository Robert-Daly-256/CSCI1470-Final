# test_story.py
import os
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from story_generation_model import TransformerStoryGenerator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        required=True,
        help="Path to folder containing preprocessed .pt files"
    )
    p.add_argument(
        "--vocab",
        required=True,
        help="Path to vocab_story.pkl"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pt checkpoint (e.g. ckpts/best.pt)"
    )
    p.add_argument(
        "--bs",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    p.add_argument(
        "--generate",
        action="store_true",
        help="If set, will generate sample stories instead of computing loss"
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt to use when generating (only if --generate is set)"
    )
    p.add_argument(
        "--max_gen_len",
        type=int,
        default=100,
        help="Maximum length of generated story"
    )
    return p.parse_args()

def load_split(data_dir, split):
    src = torch.load(os.path.join(data_dir, f"{split}_wp_source.pt"))
    tgt = torch.load(os.path.join(data_dir, f"{split}_wp_target.pt"))
    return TensorDataset(src, tgt)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            inp, lbl = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, inp)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                lbl.reshape(-1)
            )
            total_loss += loss.item()
    return total_loss / len(loader)

def generate_story(model, vocab, prompt, max_len, device):
    # Tokenize prompt using your Vocab
    tokens = prompt.lower().split()
    input_ids = vocab.encode_sentence(tokens)  # returns List[int] of length max_story_length
    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = src
    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            inp = generated[:, :-1]
            logits = model(src, inp)
            next_id = logits[0, -1].argmax().unsqueeze(0)
            generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
            if next_id.item() == vocab.end_index:
                break
    
    # Decode
    decoded = vocab.decode_sentence(generated.squeeze().cpu())
    return " ".join(decoded)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab = pickle.load(open(args.vocab, "rb"))
    vocab_size = len(vocab)

    # Instantiate model (must match training config!)
    model = TransformerStoryGenerator(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_ff=2048,
        dropout=0.1,
        max_len=512
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    if args.generate:
        story = generate_story(
            model,
            vocab,
            args.prompt,
            args.max_gen_len,
            device
        )
        print("\n=== GENERATED STORY ===\n")
        print(story)
    else:
        # Evaluate on the test split
        test_ds = load_split(args.data_dir, "test")
        test_ld = DataLoader(test_ds, batch_size=args.bs)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.padding_index)
        test_loss = evaluate(model, test_ld, criterion, device)
        print(f"Test loss (cross-entropy): {test_loss:.4f}")

if __name__ == "__main__":
    main()
