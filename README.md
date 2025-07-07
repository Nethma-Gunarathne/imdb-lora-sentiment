# imdb-lora-sentiment
Fine-tuning DistilBERT with LoRA for IMDb sentiment classification using Hugging Face Transformers and PEFT.

### Use case

- **Task:** Binary sentiment classification (positive / negative)
- **Dataset:** IMDb movie reviews (25,000 train, 25,000 test)
- **Model:** DistilBERT with LoRA

### Techniques Used
- Hugging Face Transformers + PEFT
- LoRA (r=8, alpha=32, dropout=0.1)
- Tokenization with truncation (max_length=256), padding
- Trainer API with early stopping, best model saving

---

### Performance
| Metric | Value |
|---------|--------|
| Final validation accuracy | 90.46% |
| Final test accuracy | 90.17% |
| Trainable parameters | ~1% of full model |

---

### Challenges
- Manually specifying LoRA target modules for DistilBERT (`q_lin`, `v_lin`)
- Balancing batch size with GPU memory limits in Colab

## How to Run
1️⃣ Install requirements:
```bash
pip install transformers datasets peft accelerate evaluate
