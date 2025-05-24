# IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context

This repository provides the dataset and tools to evaluate language models on our bias evaluation framework using the IndiBias dataset. 

## Overview

1. **Dataset**  
   *IndiCASA* is the dataset we contribute to the community. It contains contextually aligned stereotype and anti-stereotype sentences across five major bias dimensions relevant to the Indian context: caste, religion, disability, gender, and socioeconomic status.


2. **Encoder Training**  
   The `training_encoder` directory contains scripts for training various transformer-based text encoders using contrastive learning techniques. The primary goal is to fine-tune these encoders to produce embeddings that can effectively distinguish between similar (stereotypical) and dissimilar (anti-stereotypical) text pairs. The training process involves using different contrastive loss functions and evaluating the model's performance based on cosine similarity metrics between positive and negative pairs.

3. **Language Model Benchmarking**  
   Once the datasets are ready, the provided language model can be benchmarked using the fairness metric framework. The script defined in BiasMetric pyhandles:
   - Generation of responses by replacing `<MASK>` tokens.
   - Extraction and validation of predicted sentences.
   - Computation of sentence embeddings and cosine similarity.
   - Calculation of stereotyping and Bias score metrics.

   The complete evaluation workflow can be executed by running the `benchmark.py` script.