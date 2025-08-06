
# Multilingual Text Detoxification

This project addresses the challenge of online toxicity by transforming toxic comments into non-toxic ones while maintaining the original meaning, fluency, and grammatical correctness. It was developed as part of the [TextDetox 2024](https://github.com/pan-webis-de/pan-code/tree/master/clef24/text-detoxification) challenge. The system supports multilingual detoxification, focusing on English and Russian.


## Overview

Our approach combines baseline and advanced natural language processing models, such as backtranslation methods, BART, T5, and GPT-2 architectures. We evaluate the models on their ability to reduce toxicity while preserving the content and grammatical integrity of the input text.

---

## Task Description

Text detoxification involves:
1. **Style Transfer**: Reducing toxicity by replacing harmful or aggressive expressions with non-toxic ones.
2. **Content Preservation**: Ensuring the rephrased text retains the semantic meaning of the original.
3. **Grammatical Correctness**: Maintaining fluency and readability in the generated output.

---

## Datasets

We use the official ParaDetox datasets for English and Russian. Each comment pair includes a toxic input and its corresponding detoxified version.

- **English Dataset:** 17,769 training pairs, 1,975 validation pairs.
- **Russian Dataset:** 11,090 training pairs, 1,116 validation pairs.


| Language | Split          | Size   | Mean Length | Max | Min |
|----------|----------------|--------|-------------|-----|-----|
| English  | Train (toxic)  | 17,769 | 11.85       | 20  | 1   |
| English  | Valid (toxic)  | 1,975  | 12.00       | 20  | 5   |
| English  | Dev (toxic)    | 400    | 11.96       | 24  | 4   |
| Russian  | Train (toxic)  | 11,090 | 10.37       | 28  | 1   |
| Russian  | Valid (toxic)  | 1,116  | 10.34       | 20  | 5   |
| Russian  | Dev (toxic)    | 400    | 10.49       | 25  | 4   |

---

## File Structure

- **data/**: Training, validation, and development datasets.
- **output/**: Model-generated detoxified outputs.
- **scripts/**: Includes baseline and fine-tuning scripts.
  - `backtranslation_baseline.py`
  - `finetune_baseline.py`, `finetune_condBERT.py`
  - `evaluate.py`, `gpt2.ipynb`

---

## Baseline and Fine-Tuned Models

### Baseline: Backtranslation
- Translation → Detoxification → Backtranslation
- Uses HuggingFace: `facebook/nllb-200-distilled-600M`, `s-nlp/bart-base-detox`, `s-nlp/ruT5-base-detox`

### Fine-Tuned Models
- **BART (English)** and **T5 (Russian)** fine-tuned for multiple epochs with gradient clipping and early stopping.
- Results:
  - English BART fine-tuned improved STA and CHrF.
  - Russian T5-2 outperformed baseline in STA and SIM.

### GPT-Based Models
- GPT2LMHeadModel applied for both languages.
- Demonstrated superior CHrF performance but with trade-offs in content fidelity (SIM).

---

## Evaluation Metrics

1. **Style Transfer Accuracy (STA)** – Uses XLM-RoBERTa for toxicity classification.
2. **Content Preservation (SIM)** – Cosine similarity of LaBSE embeddings.
3. **Fluency (ChrF)** – Character and word n-gram F-scores.

Final score per sample is computed as: `J = mean(STA * SIM * ChrF)`

---

## Results

| Model             | STA     | SIM     | ChrF     | J Score  |
|------------------|---------|---------|----------|----------|
| Baseline BART     | 0.872   | 0.861   | 0.797    | 0.610    |
| Finetuned BART    | 0.879   | 0.842   | 0.835    | 0.625    |
| GPT-2 (English)   | 0.874   | 0.813   | 0.994    | 0.708    |
| Baseline T5       | 0.887   | 0.818   | 0.750    | 0.556    |
| Finetuned T5-2    | 0.887   | 0.818   | 0.750    | 0.556    |
| GPT-2 (Russian)   | 0.894   | 0.780   | 0.867    | 0.600    |

---

## Error Analysis Highlights

- **Successes**: Removed slurs or obscenity without altering the message.
- **Issues**:
  - Failure to detect toxicity.
  - Loss of expressiveness (e.g., emotion or sarcasm).
  - Over-sanitization resulting in unnatural phrasing.
  - Artefacts (e.g., "ifyify::::::ify").

---

## Future Directions

- Apply ruGPT-3.5 and explore context-aware architectures.
- Experiment with post-processing filters to fix grammatical inconsistencies.
- Fine-tune generation parameters (`top_p`, `top_k`, `temperature`) to improve expressiveness.

---

## Setup Instructions

```bash
conda env create -f speech_sanitizers_env.yml
conda activate textdetox
```

---


Read the whole report here: [Progress_report.pdf](https://github.com/spencerfliao/multilingual-text-detox/blob/9cb941a90b48fdad1226a596781d8988361428d8/Progress_report.pdf)
<img width="595" height="841" alt="Progress_report" src="https://github.com/user-attachments/assets/175a03fe-d704-474e-99b4-cfcdccc09bee" />

## Acknowledgments

Thanks to:
- TextDetox 2024 organizers
- University of British Columbia
- Contributors: Minsi Lai, Chenxin Wang, Jingyi Liao, Fangge Liao

## License

This project is licensed under the MIT License – see the LICENSE file for details.
