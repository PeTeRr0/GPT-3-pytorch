#@title Step 3: GPT-3 Training

# --- Instantiate GPT-3 model with config ---
model = Transformer(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    num_heads=config.n_heads,
    num_layers=config.n_layers,
    d_ff=config.d_ff,
    dropout=config.dropout,
    max_len=max_seq_len  
).to(config.device)

# A100 optimization
if config.compile_model:
    model = torch.compile(model)  # PyTorch 2.0+ compile optimization

# Mixed precision scaler
scaler = GradScaler(device='cuda') if config.mixed_precision else None

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)

# --- Training Loop ---
epochs = 5
model.train()
for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    total_loss = 0
    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(config.device, non_blocking=True)
        target_ids = target_ids.to(config.device, non_blocking=True)

        # Change to mixed precision forward pass
        with autocast(device_type='cuda', enabled=config.mixed_precision):
            logits = model(input_ids)
            # --- Range check --- 
            if target_ids.max().item() >= config.vocab_size or target_ids.min().item() < 0:
              print(f"[ERROR] target_ids out of range: min={target_ids.min().item()}, max={target_ids.max().item()}, vocab_size={config.vocab_size}")
              raise ValueError("invalid indices in target_ids.")
            # -------------------
            loss = criterion(logits.view(-1, config.vocab_size), target_ids.view(-1))
            loss = loss / config.gradient_accumulation_steps  # Division for gradient accumulation

        # Modify backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Add gradient accumulation step
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()  # Clear cache
            gc.collect()   # Run Python garbage collectio

        total_loss += loss.item() * config.gradient_accumulation_steps  # Recover actual loss
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

print("GPT-3 training loop completed.")

