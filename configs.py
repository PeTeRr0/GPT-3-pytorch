#@title GPT-3 Config Setup 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT3Config:
    def __init__(self):
        self.vocab_size = 50257   # GPT-3 tokenizer vocab size
        self.d_model = 768        # GPT-3 175B #12288
        self.n_layers = 12        # GPT-3 175B # 96
        self.n_heads = 12         # GPT-3 175B # 96
        self.d_ff = 3072          # GPT-3 175B # 49152
        self.dropout = 0.1        # no dropout in the GPT-3 paper 
        self.max_seq_len = 256    # GPT-3 max context length # 2048
        self.lr = 1e-4            # Adam lr
        self.betas = (0.9, 0.95)
        self.eps = 1e-8
        self.weight_decay = 0.0
        # A100 optimization setup
        self.gradient_accumulation_steps = 16  # Large batch simulation
        self.mixed_precision = True           # FP16/BF16
        self.compile_model = True             # torch.compile
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPT3Config()
