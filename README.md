# GPT-3: Language Models are Few-Shot Learners

Pytorch implementation of the paper [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) by OpenAI team. This paper demonstrates that scaling up autoregressive Transformer language models and training them with the simple next-token prediction objective produces powerful in-context learning—models that can perform new tasks from natural-language instructions and a few examples provided in the prompt, without any task-specific fine-tuning. It introduces GPT-3, a family of Transformer-based language models scaled up to 175 billion parameters (with a 2048-token context window) and trained on a large mixture of web text, books, and Wikipedia. The authors show that performance on a wide range of tasks (translation, question answering, cloze, commonsense and reasoning benchmarks, etc.) improves substantially with model scale, enabling strong zero-, one-, and few-shot capabilities where careful prompt design replaces the need for additional task-specific parameters. The paper also evaluates failure modes and tradeoffs—data contamination and memorization risks, social biases inherited from web-scale training corpora, and the significant compute and environmental costs of training such massive models.

## Dataset Preparation

This project utilizes the WMT14 dataset, a widely used benchmark for machine translation research. Specifically, the French-to-English translation subset is employed as the source text data.

The dataset provides parallel sentence pairs and is commonly used for training and evaluating language models in translation and general natural language understanding tasks. A random sampling strategy is applied to select English or French sentences from the dataset for model training.

To process the text data, the GPT-3 Byte-Pair Encoding (BPE) tokenizer ("cl100k_base") is employed. This tokenizer converts raw text into token sequences compatible with GPT-3 style language models.

## Learning Overview in GPT-3 Pre-training
![figure1](assets/figure1.png)
### **Outer Loop**
- Represents the overall pre-training process.
- Learning is performed **via Stochastic Gradient Descent (SGD)** across a large dataset in an **unsupervised manner**.
- This loop is responsible for adjusting the model parameters over time to minimize prediction error across many sequences.

### **Inner Loop**
- Represents **in-context learning** within individual sequences.
- The model processes sequences of examples where it can make predictions based on the context provided **without explicit parameter updates**.
- Each inner loop corresponds to a specific sequence of examples:

#### Sequence #1
- Example task: arithmetic corrections
#### Sequence #2
- Example task: spelling corrections
#### Sequence #3
- Example task: language translation

## GPT-3 Learning Performance
![figure2](assets/figure2.png)
### **Axes**
- **X-axis**: Number of examples provided in the context (*K*).
- **Y-axis**: Accuracy (%) on the evaluation task.

### **Model Sizes**
- **175B Parameters** (blue)
- **13B Parameters** (orange)
- **1.3B Parameters** (green)

### **Prompting Styles**
- **Natural Language Prompt** (solid line): A descriptive instruction or example.
- **No Prompt** (dashed line): The task is given without any introductory instruction.

### **Key Observations**
1. **Zero-shot (0 examples)**  
   - The largest model (175B) shows high baseline accuracy (8%) without examples.
   - Smaller models start near 0%, indicating limited zero-shot ability.

2. **One-shot (1 example)**  
   - Adding just one example greatly boosts accuracy for large models (175B jumps to 45%).
   - Mid-size (13B) and small (1.3B) models show smaller gains.

3. **Few-shot (10 - 100 examples)**  
   - Accuracy increases steadily with more examples in context.
   - The 175B model approaches 67% accuracy with enough examples, showing strong in-context learning.
   - The gap between "Natural Language Prompt" and "No Prompt" narrows as more examples are given.

4. **Scaling Effect**  
   - Larger models benefit significantly more from few-shot prompting.
   - Smaller models (1.3B) barely improve, suggesting that in-context learning ability scales with parameter count.
=============================================
## Aggregate Performance Across Benchmarks
![figure3](assets/figure3.png)
### **Axes**
- **X-axis**: Model size in billions of parameters (from 0.1B to 175B).
- **Y-axis**: Accuracy (%) across all evaluated benchmarks.

### **Key Observations**
   **Scaling Improves Performance**
   - All three modes (few-shot, one-shot, zero-shot) improve steadily as model size increases.
   - Few-shot consistently outperforms one-shot and zero-shot across all model sizes.
   - The largest model (175B) achieves:
     - **Few-Shot**: ~58% accuracy
     - **One-Shot**: ~51% accuracy
     - **Zero-Shot**: ~42% accuracy

## In-Context Learning vs. Traditional Fine-Tuning
![figure4](assets/figure4.png)
### **In-Context Learning (No Parameter Updates)**
The model answers based on the task description and provided examples **within the prompt**.  
No gradient updates are performed during inference.

#### **Zero-Shot**
- **Description**: Only the task description is given, no examples.
- The model infers the answer solely from its pre-trained knowledge.

#### **One-Shot**
- **Description**: Task description + one example.
- The single example helps guide the model’s translation.

#### **Few-Shot**
- **Description**: Task description + multiple examples (10 - 100 examples).
- - Multiple examples allow the model to better understand the pattern before answering.

---

### **Traditional Fine-Tuning (Not Used for GPT-3)**
- The model is trained with **repeated gradient updates** on a large set of labeled examples for a specific task.
- Process:
1. Input an example (`sea otter => loutre de mer`)
2. Compute loss and apply a gradient update.
3. Repeat for all training examples.
4. After fine-tuning, the model is specialized for that task.
- GPT-3 skips this step and relies solely on **in-context learning** during evaluation.

## Scaling Laws: Validation Loss vs. Compute
![figure5](assets/figure5.png)
### **Axes**
- All curves follow a **power-law relationship** between compute and loss.
- **X-axis**: Compute measured in **PetaFLOP/s-days** (log scale).
- **Y-axis**: Validation loss (lower is better).
- **Color Gradient**: Number of model parameters (**10⁵** (purple) to **10¹¹** (yellow)).

---

### **Observations**
1. **Consistent Scaling Behavior**
   - Across all parameter sizes, increasing compute reduces validation loss.

2. **Larger Models Require More Compute**
   - Small models (purple/blue) reach optimal loss with relatively little compute.
   - Large models (green/yellow) need **massively more compute** to fully utilize their capacity.

## **GPT-3 Training Curves
![figure6](assets/figure6.png)
## **Axes
**X-axis**: Tokens elapsed (billions).  
**Y-axis**: Cross-entropy loss (nats/token, smoothed).  
**Color gradient:** Number of model parameters (**10⁸** to (purple) **10¹¹** (yellow)).  

### **Observations**
**1. Early rapid improvement**  
- Loss drops steeply in the initial phase (first ~20B tokens).

**2. Diminishing marginal gains**  
- After the sharp early decrease, loss continues to fall but more slowly—each additional billion tokens yields smaller improvements, especially for smaller models.

**3. Train vs. validation gap**  
- Validation loss is consistently slightly higher than training loss. The gap is larger for smaller models, indicating relatively weaker generalization.

## Configuration
```python
class GPT3Config:
    def __init__(self):
        self.vocab_size = 50257     # Size of the GPT-3 tokenizer vocabulary
        self.d_model = 768          # Model hidden dimension (GPT-3 175B uses 12288)
        self.n_layers = 12          # Number of Transformer decoder layers (GPT-3 175B uses 96 layers)
        self.n_heads = 12           # Number of attention heads per layer (GPT-3 175B uses 96 heads)
        self.d_ff = 3072             # Feed-forward network hidden dimension (GPT-3 175B uses 49152)
        self.dropout = 0.1           # Dropout rate (GPT-3 paper did not use dropout)
        self.max_seq_len = 512       # Maximum sequence length (GPT-3 uses up to 2048 tokens)
        self.lr = 1e-4                # Learning rate for Adam optimizer
        self.betas = (0.9, 0.95)     # Beta values for Adam optimizer
        self.eps = 1e-8              # Epsilon for numerical stability in Adam optimizer
        self.weight_decay = 0.0      # Weight decay for regularization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
                                     # Automatically select GPU if available, else use CPU
```

## Counting French Words in Dataset Samples

This analysis inspects the frequency of common French keywords within a subset of dataset texts, providing insights into the multilingual nature of the dataset used for training.

### Methodology
- A list of frequent French keywords (e.g., "le", "la", "et", "de", "bonjour", "merci", "oui") is defined.
- The first 100,000 text samples from the dataset are examined.
- Each text is wrapped with spaces to detect whole word matches accurately.
- The script counts occurrences of each French keyword and tallies total keyword hits, including multiple hits from the same text.

### Results
- The sample size analyzed consists of 100,000 texts.
- Keyword counts show notable presence of French terms, such as "de" (27,004 occurrences), "la" (20,810), "que" (17,701), "et" (17,140), and "à" (16,463).
- The total French keyword occurrences counted are 125,710, indicating multiple keywords often appear in a single text.

### Training Progress Context
The dataset was used to train a GPT-3 style model over 5 epochs. Training logs indicate steady loss reduction across epochs:
- Epoch 1 loss: 4.62
- Epoch 2 loss: 3.6
- Epoch 3 loss: 3.2
- Epoch 4 loss: 2.98
- Epoch 5 loss: 2.83

## Zero-Shot Testing Results and Future Directions

### Zero-Shot Text Generation

A zero-shot evaluation was performed to test the model’s ability to generate French translations given English prompts using a sampling-based decoding method with top-k filtering.

**Test Prompts:**
- "English: Good morning.\nFrench:"
- "English: Thank you for your help.\nFrench:"
- "English: I enjoy learning new things.\nFrench:"

**Observations:**
- The generated outputs failed to produce coherent or relevant French translations.
- Instead, the outputs were mostly nonsensical, random token sequences, indicating the model has not effectively learned the translation or language generation task.

### Training Context and Limitations

- The model was trained for 5 epochs on a limited portion of the WMT14 dataset.
- Training loss decreased steadily, but the final performance does not yet reflect practical language generation capabilities.

### Future Work

These results highlight the need for training on a much larger and more diverse dataset to improve language understanding and generation. Expanding the dataset size and training time would enable the model to better capture language patterns and semantics, essential for tasks like translation and zero-shot generation.

Increasing dataset scale, potentially incorporating various multilingual corpora and larger token counts, will be critical for achieving meaningful and coherent generative behavior in GPT-3 models.

### Test Prompts:
- "English: Good morning.\nFrench:"
- "English: Thank you for your help.\nFrench:"
- "English: I enjoy learning new things.\nFrench:"

### Observations:
- Despite the one-shot context, the model repeatedly outputs the example translation phrase instead of producing meaningful translations for the new prompts.
- The generated continuations devolve into incoherent, nonsensical token sequences unrelated to the intended French translations.
- This indicates the model is unable to generalize from the single demonstration and generate valid translations given the current training state.

### Summary

The one-shot testing results align with zero-shot findings, demonstrating that the model’s current training state is insufficient for producing coherent language generation or translation in this scenario.

## Few-Shot Testing Results

### Test Prompts:
- "English: Good morning.\nFrench:"
- "English: Thank you for your help.\nFrench:"
- "English: I enjoy learning new things.\nFrench:"

### Observations:
- Despite the improved prompting strategy with multiple examples, generation quality remains poor.
- The model continues to repeat the example examples verbatim and produces largely unintelligible or irrelevant continuations.
- No meaningful or accurate French translations are generated for the new test prompts.

### Summary

The few-shot setting provides more contextual clues but is still insufficient for the current model to produce coherent outputs. This indicates the model requires further training and exposure to larger, higher-quality datasets to enable effective few-shot learning and generalization.

Ongoing improvements in dataset scale, model capacity, and training duration will be necessary to achieve the level of performance demonstrated by state-of-the-art GPT-3 models.

## Final Remarks

This project presents a PyTorch implementation inspired by GPT-3, demonstrating key concepts from the original "Language Models are Few-Shot Learners" paper. The work includes dataset preparation, training on the WMT14 translation dataset, and evaluations under zero-shot, one-shot, and few-shot scenarios.

Although training loss steadily decreased over 5 epochs, the current model struggles to generate coherent and relevant French translations, as evidenced by the zero-, one-, and few-shot tests. These results highlight the challenges of scaling language models and the critical importance of large, diverse training data to achieve strong in-context learning and generation capabilities.

Future work will focus on training with substantially larger datasets and extended compute resources to better approximate the performance characteristics reported by the original GPT-3 model, enabling the development of more robust few-shot learning abilities.
