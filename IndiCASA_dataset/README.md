# IndiCASA Dataset Documentation

## 1. Overview

The IndiCASA (Indian Contextual Representation of Algorithmic Stereotypes) dataset is a comprehensive collection of stereotype and anti-stereotype sentence pairs across five major bias dimensions relevant to the Indian context: caste, religion, disability, gender, and socioeconomic status. This dataset provides researchers with rich contextual data to study, measure, and mitigate social biases in language models and other AI systems.

## 2. Dataset Structure

IndiCASA is organized as a `DatasetDict` with five keys corresponding to each bias type:

```
IndiCASA/
├── caste/
├── religion/
├── disability/
├── gender/
└── socioeconomic/
```

Each bias category contains parallel lists of stereotype and anti-stereotype sentences organized by context groups, allowing researchers to analyze bias manifestations across different social dimensions.

## 3. Dataset Schema

Each bias-specific dataset contains the following fields:

| Field                | Type   | Description                                                        |
|----------------------|--------|--------------------------------------------------------------------|
| `context_id`         | int    | Context group identifier                                           |
| `sentence`           | str    | Stereotype or anti-stereotype sentence                             |
| `type`               | str    | "stereotype" or "anti_stereotype"                                  |
| `annotator1_rating`  | int    | Annotator 1's rating (1-5)                                         |
| `annotator2_rating`  | int    | Annotator 2's rating (1-5)                                         |

## 4. Annotation Process

Each sentence was **independently rated by two annotators** to validate its label (stereotype or anti-stereotype) using the following 5-point Likert scale:

- **1 = Totally disagree** (Does NOT reflect the stereotype or anti-stereotype in the given domain)
- **2 = Disagree**
- **3 = Neutral**
- **4 = Agree**
- **5 = Totally Agree** (Accurately reflects the stereotype or anti-stereotype in the given domain)

This process ensures both the **quality** and **validity** of the dataset for downstream research.
      
---

## 5. Loading and Accessing the Dataset

### Installation

```bash
pip install datasets pandas
```

### Loading

```python
from datasets import load_from_disk

# Load original grouped format
IndiCASA = load_from_disk("./hf_datasets/IndiCASA")

# Load annotated flat format
IndiCASA_annotated = load_from_disk("./hf_datasets/IndiCASA_Annotated")
```

### Example: Accessing Context Pairs (Original Format)

```python
example = IndiCASA["caste"][0]
print("Stereotypes:")
for i, stereo in enumerate(example["stereotypes"]):
    print(f"{i+1}. {stereo}")
print("\nAnti-stereotypes:")
for i, anti in enumerate(example["anti_stereotypes"]):
    print(f"{i+1}. {anti}")
```

### Example: Accessing Annotated Sentences

```python
row = IndiCASA_annotated["caste"][0]
print(row["sentence"], row["type"], row["annotator1_rating"], row["annotator2_rating"])
```

### Example: Pandas DataFrame Conversion

You can convert annotated splits to pandas for flexible analysis:

```python
import pandas as pd
caste_df = IndiCASA_annotated["caste"].to_pandas()
# Group by context_id to reconstruct context pairs if needed
grouped = caste_df.groupby('context_id')
for context_id, group in grouped:
    stereotypes = group[group['type'] == 'stereotype']['sentence'].tolist()
    anti_stereotypes = group[group['type'] == 'anti_stereotype']['sentence'].tolist()
    # ... use as needed
```

## 6. Research Applications

The IndiCASA dataset supports various research applications, including:

- Evaluating bias in language models trained on Indian corpora
- Developing bias mitigation techniques for NLP systems
- Comparative analysis of biases across different social dimensions
- Creating fairness benchmarks for responsible AI development
- Studying the linguistic patterns of stereotypes in Indian contexts

<!-- ## 7. Dataset Citation

When using the IndiCASA dataset in your research, please cite:

```
@dataset{IndiCASA,
  title = {IndiCASA: Indian Contextual Representation of Algorithmic Stereotypes},
  author = {Anonymous},
  year = {2025},
  url = {},
}
``` -->

## 7. Ethical Considerations

This dataset contains explicit stereotype content for research purposes. Researchers should acknowledge that:

1. The inclusion of stereotypical content does not endorse these views
2. Care should be taken when deploying models trained or evaluated on this data
3. Applications built using this dataset should implement appropriate bias mitigation strategies

By providing contextually aligned stereotype and anti-stereotype examples, IndiCASA enables more robust evaluation and mitigation of harmful biases in AI systems relevant to Indian contexts.



Can you please write an updated redame with all the new updates, I have with respect to annoatations and new format of daat and finally the cohen kappa thing