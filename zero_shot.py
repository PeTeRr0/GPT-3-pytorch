#@title GPT-3 Zero-Shot Testing

def generate_text(prompt, max_new_tokens=50, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=config.device).unsqueeze(0)

    tgt_mask = look_ahead_mask_(input_ids.size(1), device=config.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids, tgt_mask)
            next_token_logits = logits[:, -1, :]

            # top-k filtering
            values, indices = torch.topk(next_token_logits, k=top_k)
            probs = torch.zeros_like(next_token_logits).scatter_(1, indices, values)
            probs = F.softmax(probs, dim=-1)

            # Sampling
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            tgt_mask = look_ahead_mask_(input_ids.size(1), device=config.device)

            # EOS token processing (0-padding so that eos token definition necessary)
            if next_token_id.item() == tokenizer.eot_token:
                break

    return tokenizer.decode(input_ids[0].tolist())

# Zero-shot Evaluation
zero_shot_prompts = [
    "English: Good morning.\nFrench:",
    "English: Thank you for your help.\nFrench:",
    "English: I enjoy learning new things.\nFrench:"
]

for i, prompt in enumerate(zero_shot_prompts, start=1):
    output = generate_text(prompt)
    print(f"[Test {i}]")
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}\n")