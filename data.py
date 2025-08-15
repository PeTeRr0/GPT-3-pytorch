#@title GPT-3 Dataset Preparation

class GPT3Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len, vocab_size=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size or tokenizer.n_vocab

        # Convert text into a list of tokens using a tokenizer
        self.tokens = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            self.tokens.extend(token_ids)

        # total number of sequences
        self.num_sequences = len(self.tokens) // max_seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        seq = self.tokens[start:end]

        # Apply 0-padding if the sequence length is insufficient
        if len(seq) < self.max_seq_len:
            seq += [0] * (self.max_seq_len - len(seq))

        # Ensure token IDs do not exceed vocab_size - 1
        seq = [max(0, min(t, self.vocab_size - 1)) for t in seq]



        input_ids = torch.tensor(seq[:-1], dtype=torch.long)    # input
        target_ids = torch.tensor(seq[1:], dtype=torch.long)    # next token
        return input_ids, target_ids

# --- Load dataset (WMT14) ---
def sample_translation(example):
    return {"text": example['translation']['en'] if random.random() >= 0.4 else example['translation']['fr']}

dataset_wmt14 = load_dataset("wmt14", "fr-en", split="train")
texts = dataset_wmt14.map(sample_translation, batched=False)['text'][:1300000]  # Number of samples

print(f"Total texts loaded: {len(texts)}")

# --- GPT-3 tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3 BPE tokenizer
print(f"Tokenizer vocab size: {tokenizer.n_vocab}")

# Update config vocab_size to match tokenizer
config.vocab_size = tokenizer.n_vocab
print(f"Updated config vocab_size to: {config.vocab_size}")

# --- Dataset & DataLoader ---
max_seq_len = 256  # A100 env
train_dataset = GPT3Dataset(texts, tokenizer, max_seq_len=max_seq_len, vocab_size=config.vocab_size)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

print(f"Total sequences: {len(train_dataset)}")
print(f"Example input_ids shape: {train_dataset[0][0].shape}")
