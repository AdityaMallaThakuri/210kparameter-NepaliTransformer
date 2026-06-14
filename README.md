# 210k Parameter Nepali Transformer

A character-level GPT trained on Nepali poetry — built from scratch using PyTorch.

---

## What This Is

A decoder-only transformer (GPT architecture) trained on Laxmi Prasad Devkota's Nepali poems. 

**Model size:** 210,000 parameters  
**Training data:** 119,162 characters of Devkota's poetry  
**Vocab size:** 74 unique Devanagari characters  
**Final train loss:** 1.97 | **Final val loss:** 2.35  

---

## Sample Output

After 5000 training steps the model generates Nepali text in Devkota's style:

```
आत्माले झुलेको आँखामा गिगोरि खिरण छ,
लाई कसरी !
गराचुरा घार भन्दामा तिम्ता बादलान,
मृँधि ए पुविएने फूलेर घाम ढुलबुल,
तिमौलाङ्गीतसँग,
```

Real Nepali words and grammatical suffixes (-ले, -मा, -को, -छ) emerge from training.

---

## Architecture

```
Input characters
      ↓
Token embedding (64 dim) + Positional embedding (64 dim)
      ↓
Transformer Block × 4
  ├── LayerNorm
  ├── Multi-Head Self-Attention (4 heads × 16 dim)
  ├── Residual connection
  ├── LayerNorm
  ├── FeedForward (64 → 256 → 64)
  └── Residual connection
      ↓
LayerNorm
      ↓
Linear → 74 logits → Softmax → next character
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| block_size | 32 |
| n_embd | 64 |
| n_head | 4 |
| n_layer | 4 |
| batch_size | 16 |
| max_iters | 5000 |
| learning_rate | 1e-3 |
| dropout | 0.0 |

---

## How To Run

### 1. Get the training data
Download Devkota's poems from [this GitHub repo](https://github.com/de-sawal/Poem-Generator/blob/master/lspd.txt) and save as `lspd.txt`.

### 2. Open the notebook in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/210k-parameter-nepali-transformer/blob/main/nepali_gpt_devkota.ipynb)

### 3. Upload lspd.txt
Upload `lspd.txt` to the Colab session files.

### 4. Run all cells
Training takes ~10 minutes on CPU, ~2 minutes on GPU.

---

## Why This Matters for Lost Voices

Nepal has 124 languages — 11 already extinct, 37 with fewer than 1,000 speakers. **Lost Voices** is building AI pipelines to preserve these languages before the last speakers are gone.

This notebook is a baseline — proving the pipeline works on Nepali before applying it to lower-resource languages like **Tamang** and **Sunuwar**.

The full Lost Voices stack:
- **NLLB-200** fine-tuned on NepTam corpus → Nepali ↔ Tamang translation
- **Whisper** with LoRA → speech recognition for recorded audio
- **Coqui TTS / VITS** → synthetic voice generation

Every concept in this notebook (embeddings, attention, residuals, cross-entropy loss, AdamW) is inside each of those models.

---

## Requirements

```
torch>=2.0.0
```

No other dependencies — pure PyTorch.

---

## References

- Karpathy, A. (2023). [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Devkota, L.P. Nepali poems corpus via [de-sawal/Poem-Generator](https://github.com/de-sawal/Poem-Generator)


---

