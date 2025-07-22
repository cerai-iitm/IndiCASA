# IndiCASA: Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context

This directory, `BiasMetric`, is a core component of the IndiCASA project. It provides the tools and scripts necessary to evaluate bias in Language Models (LLMs) using the IndiBias dataset, and encoder tuned using IndiCASA dataset, with a particular focus on the Indian socio-cultural context. The framework measures stereotyping and fairness through metrics derived from model-generated text, sentence embeddings, and their cosine similarities.

## Repository Structure

The `BiasMetric` folder is organized as follows:

```
BiasMetric/
├── benchmark.py                # Main script to run the bias benchmarking on an LLM.
├── BiasMetric.py               # Core Python module defining the StereoEncoder and BiasMetric classes.
├── Masked_IndiBias.csv         # Dataset with masked sentences for LLM prediction tasks.
├── Prepare_IndiBias.py         # Script to download and preprocess the raw IndiBias dataset.
├── README.md                   # This README file.
├── requirements.txt            # Python dependencies for this component.
└── best_contrastive_model/     # Directory containing the pre-trained sentence encoder.
    ├── best_model.pth          # Weights of the fine-tuned sentence encoder.
    └── Description.txt         # Training details for the best_model.pth.
```

## Key Components

### 1. Dataset Preparation

-   **Prepare_IndiBias.py**: This script automates the downloading of the raw IndiBias dataset from its source. It then cleans and transforms the data, aligning stereotype and anti-stereotype sentence pairs.
    -   **Output**: The script saves the processed data to a file named IndiBias_Aligned_Stereotype.csv in the directory from which the script is executed.
    -   **Note**: The `BiasMetric.py` script expects this file at `../IndiBias_Aligned_Stereotype.csv`.

-   **`Masked_IndiBias.csv`**: This CSV file contains sentences from the IndiBias dataset where specific tokens have been replaced with `<MASK>`. These masked sentences are used as prompts for the LLM being evaluated. The `BiasMetric.py` script loads this file from its local directory (`./Masked_IndiBias.csv`).

### 2. Stereotype Encoder

-   **`StereoEncoder` class (in `BiasMetric.py`)**: This PyTorch module is responsible for generating sentence embeddings.
-   **`best_contrastive_model/`**: This directory houses the pre-trained sentence encoder model fine-tuned to be sensitive to stereotypical nuances in text.
    -   **`best_model.pth`**: The saved weights of the fine-tuned transformer-based sentence encoder. The `StereoEncoder` loads this model from `./best_contrastive_model` (relative to `BiasMetric.py`).
    -   **`Description.txt`**: A text file detailing the training parameters (e.g., loss function, learning rate, epochs) used to produce `best_model.pth`. This model is likely trained using the scripts in the main project's `training_encoder` directory.

### 3. Bias Evaluation Core (BiasMetric.py)

-   **`BiasMetric` class (in `BiasMetric.py`)**: This class orchestrates the entire bias evaluation pipeline.
    -   **Initialization**: Takes an LLM, its tokenizer, a list of bias types to evaluate, and the computation device (`cpu` or `cuda`) as input. It loads IndiBias_Aligned_Stereotype.csv (from sant) and `Masked_IndiBias.csv` (from Predicting-Indian-Caste).
    -   **Response Generation (`generate_responses`)**: Uses the input LLM to fill in the `<MASK>` tokens in sentences from `Masked_IndiBias.csv`. It employs a specific `PROMPT_TEMPLATE` for this task.
    -   **Embedding Generation (`get_embeddings`)**: Utilizes the `StereoEncoder` (with `best_model.pth`) to compute embeddings for:
        1.  Original stereotype sentences from IndiBias_Aligned_Stereotype.csv.
        2.  Original anti-stereotype sentences from IndiBias_Aligned_Stereotype.csv.
        3.  The LLM-generated sentences (predictions for masked inputs).
    -   **Similarity Calculation (`calc_cosine_similarity`)**: Computes cosine similarities between the embeddings of LLM-generated sentences and the embeddings of the original stereotype and anti-stereotype sentences.
    -   **Metric Computation (`calculate_fairness`)**: This is the main method that runs the pipeline and calculates several bias metrics:
        -   **Stereotypical Metric**: The proportion of times the LLM's prediction is semantically closer to the stereotypical sentence than the anti-stereotypical one.
        -   **KL Divergence**: Measures how the LLM's distribution of stereotypical preference deviates from a perfectly unbiased distribution (e.g., [0.5, 0.5]).
        -   **Bias Score**: A distance-based score (derived from the Stereotypical Metric and an unbiased [0.5, 0.5] distribution) indicating the magnitude of bias.
        -   The method also saves the LLM's predictions to a CSV file (e.g., `predicted_indibias_deepseek-r1-llama.csv` in the current working directory).

### 4. Benchmarking Script (benchmark.py)

-   **`benchmark.py`**: This is the command-line entry point to run the full bias evaluation pipeline on a specified LLM.
    -   It parses command-line arguments, primarily the Hugging Face model identifier for the LLM to be evaluated (defaults to `google/gemma-2-9b-it`).
    -   Sets up logging to both console and a file (e.g., `benchmark_logs/benchmark_<model_name>.log`).
    -   Loads the specified LLM and tokenizer.
    -   Instantiates the `BiasMetric` class.
    -   Calls the `calculate_fairness` method to get the bias metrics.
    -   Prints and logs the resulting Stereotype Metric, Bias Score, and KL Divergence.

## Setup and Installation

1.  **Prerequisites**:
    *   Python (version 3.8 or higher recommended).
    *   `pip` for installing packages.

2.  **Clone the Repository** (if you haven't already):
    ```bash
    git clone https://github.com/Matrixmang0/IndiCASA.git # Or your repository URL
    cd IndiCASA/BiasMetric # Navigate to this directory
    ```

3.  **Set up Python Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    Install the required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    The dependencies include:
    ```txt
    // filepath: GitHub/BiasMetric/requirements.txt
    transformers
    torch
    datasets
    scikit-learn
    numpy
    pandas
    ```

## Usage

Follow these steps to run the bias evaluation:

### Step 1: Prepare the IndiBias Dataset

Run the `Prepare_IndiBias.py` script. It's recommended to run this script from the root of the `IndiCASA` project directory (e.g., `IndiCASA/`) to ensure the output file IndiBias_Aligned_Stereotype.csv is created where `BiasMetric.py` expects it (`../IndiBias_Aligned_Stereotype.csv` relative to BiasMetric.py).

If you run it from the `IndiCASA/GitHub/BiasMetric/` directory:
```bash
python Prepare_IndiBias.py
```
This will create IndiBias_Aligned_Stereotype.csv inside the `IndiCASA/BiasMetric/` directory. 


Alternatively, navigate to the `IndiCASA/` directory and run:
```bash
python BiasMetric/Prepare_IndiBias.py
```
This will create IndiBias_Aligned_Stereotype.csv directly in the `IndiCASA/` directory.

Ensure `Masked_IndiBias.csv` is present in the `BiasMetric` directory.

### Step 2: Run Benchmarking

Execute the `benchmark.py` script from within the `BiasMetric` directory. You can specify the Hugging Face model identifier for the LLM you want to evaluate using the `--model_url` argument.

```bash
cd IndiCASA/BiasMetric/ # Ensure you are in the BiasMetric directory
python benchmark.py --model_url <your_huggingface_model_identifier>
```
For example, to evaluate `google/gemma-2-9b-it` (the default):
```bash
python benchmark.py
```
Or for another model:
```bash
python benchmark.py --model_url meta-llama/Llama-2-7b-chat-hf
```

The script will:
*   Load the specified LLM.
*   Perform the bias calculations using the `BiasMetric` framework.
*   Print the Stereotype Metric, Bias Score, and KL Divergence to the console.
*   Save detailed logs in the `benchmark_logs/` directory (created within `BiasMetric`).
*   Save the LLM's generated predictions to a CSV file (e.g., `predicted_indibias_deepseek-r1-llama.csv`) in the `BiasMetric` directory.

## Output Files

When running the benchmark, expect the following outputs:
*   **Console Output**: Metrics printed directly to your terminal.
*   **Log Files**: Detailed logs stored in `benchmark_logs/benchmark_<sanitized_model_name>.log` within the `BiasMetric` directory.
*   **Prediction CSV**: A file like `predicted_indibias_<model_specific_tag>.csv` containing the sentences generated by the LLM, saved in the `BiasMetric` directory.

## Ethical Considerations

This framework and the underlying datasets (IndiBias) involve the study of social stereotypes, which are sensitive in nature.
*   The inclusion of stereotypical content is solely for research purposes: to understand, measure, and ultimately mitigate biases in AI systems. It does not imply endorsement of these views.
*   Exercise caution when deploying or interpreting results from models trained or evaluated using this data.
*   Researchers and developers are encouraged to implement appropriate bias mitigation strategies and remain mindful of the potential societal impact of their work.

For more information on the broader IndiCASA project, refer to the main README.md in the GitHub directory.