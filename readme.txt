Project : DistilBERT with LoRA on IMDb Sentiment Classification

This project fine-tunes `distilbert-base-uncased` using Low-Rank Adaptation (LoRA) for binary sentiment classification on the IMDb dataset. 
The goal is to achieve high accuracy with parameter-efficient training.

## **Use Case**
- **Task:** Binary sentiment classification (positive / negative)
- **Dataset:** IMDb movie reviews (25,000 train, 25,000 test)
- **Model:** DistilBERT with LoRA adapters applied to `q_lin` and `v_lin`

---

## **Techniques Used**
- Hugging Face Transformers + PEFT
- LoRA (r=8, alpha=32, dropout=0.1)
- Tokenization with truncation (max_length=256), padding
- Trainer API with early stopping, best model saving

---

## **Performance**
| Metric | Value |
|---------|--------|
| Final validation accuracy | 90.46% |
| Final test accuracy | 90.17% |
| Trainable parameters | ~1% of full model |

---

## **Challenges**
- Manually specifying LoRA target modules for DistilBERT (`q_lin`, `v_lin`)
- Balancing batch size with GPU memory limits in Colab

## **How to Run**
1️⃣ Install requirements:
```bash
pip install transformers datasets peft accelerate evaluate

