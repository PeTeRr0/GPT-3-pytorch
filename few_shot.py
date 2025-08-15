#@title GPT-3 Few-Shot Testing

# Few-shot example (English -> French translation)
examples = (
    "English: Hello, how are you?\nFrench: Bonjour, comment ça va?\n\n"
    "English: I love programming.\nFrench: J'adore programmer.\n\n"
    "English: The weather is nice today.\nFrench: Il fait beau aujourd'hui.\n\n"
)

# Test
few_shot_prompts = [
    examples + "English: Good morning.\nFrench:",
    examples + "English: Thank you for your help.\nFrench:",
    examples + "English: I enjoy learning new things.\nFrench:"
]

for i, prompt in enumerate(few_shot_prompts, start=1):
    output = generate_text(prompt, max_new_tokens=50, top_k=50)
    print(f"[Test {i}]")
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}\n")
