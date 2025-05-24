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

| Field | Type | Description |
|-------|------|-------------|
| `context_id` | int | Unique identifier that groups semantically related stereotype and anti-stereotype sentences |
| `stereotypes` | List[str] | Collection of sentences representing stereotypical beliefs or portrayals about specific groups |
| `anti_stereotypes` | List[str] | Collection of sentences presenting counter-narratives to the corresponding stereotypes |

## 4. Loading and Accessing the Dataset

### Installation Requirements

```python
# Install required libraries
pip install datasets pandas
```

### Loading the Dataset

```python
from datasets import load_from_disk

# Load the complete dataset
IndiCASA = load_from_disk("./hf_datasets/IndiCASA")

# Access individual bias categories
caste_dataset = IndiCASA["caste"]
religion_dataset = IndiCASA["religion"]
gender_dataset = IndiCASA["gender"]
disability_dataset = IndiCASA["disability"]
socioeconomic_dataset = IndiCASA["socioeconomic"]
```

### Basic Dataset Exploration

```python
# View dataset structure and information
IndiCASA["caste"].column_names  # List column names
IndiCASA["caste"].features      # View column data types
IndiCASA["caste"][0]            # View first example
```

### Converting to Pandas for Analysis

```python
# Convert to pandas for more flexible data manipulation
caste_df = IndiCASA["caste"].to_pandas()

# View specific columns
caste_df[["context_id", "stereotypes", "anti_stereotypes"]].head()
```

## 5. Working with Context Pairs

Each entry contains lists of stereotype and anti-stereotype sentences within the same contextual frame:

```python
# Get a specific example
example = IndiCASA["caste"][0]

# Print stereotype/anti-stereotype pairs
print("Stereotypes:")
for i, stereo in enumerate(example["stereotypes"]):
    print(f"{i+1}. {stereo}")

print("\nAnti-stereotypes:")
for i, anti in enumerate(example["anti_stereotypes"]):
    print(f"{i+1}. {anti}")
```

## 6. Advanced Dataset Operations

### Filtering Examples

```python
# Filter examples with more than 10 stereotype sentences
filtered = IndiCASA["caste"].filter(lambda example: len(example['stereotypes']) > 10)
```

### Random Sampling

```python
# Get random examples (with reproducible seed)
random_examples = IndiCASA["caste"].shuffle(seed=42).select(range(3))
```

## 7. Research Applications

The IndiCASA dataset supports various research applications, including:

- Evaluating bias in language models trained on Indian corpora
- Developing bias mitigation techniques for NLP systems
- Comparative analysis of biases across different social dimensions
- Creating fairness benchmarks for responsible AI development
- Studying the linguistic patterns of stereotypes in Indian contexts

<!-- ## 8. Dataset Citation

When using the IndiCASA dataset in your research, please cite:

```
@dataset{IndiCASA,
  title = {IndiCASA: Indian Contextual Representation of Algorithmic Stereotypes},
  author = {Anonymous},
  year = {2025},
  url = {},
}
``` -->

## 9. Ethical Considerations

This dataset contains explicit stereotype content for research purposes. Researchers should acknowledge that:

1. The inclusion of stereotypical content does not endorse these views
2. Care should be taken when deploying models trained or evaluated on this data
3. Applications built using this dataset should implement appropriate bias mitigation strategies

By providing contextually aligned stereotype and anti-stereotype examples, IndiCASA enables more robust evaluation and mitigation of harmful biases in AI systems relevant to Indian contexts.