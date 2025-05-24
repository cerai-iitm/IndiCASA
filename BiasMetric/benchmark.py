"""
This script calculates bias and fairness metrics for a given language model.
It sets up logging, loads the tokenizer and model, and computes metrics such as stereotype 
and KL divergence for various bias types.

Usage:
  python benchmark.py --model_url <model_identifier>
  
If no model URL is provided, a default model will be used.
"""

import os
import logging
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from BiasMetric import BiasMetric


def setup_logger(log_dir="benchmark_logs", log_filename="training.log"):
  """
  Sets up logging for the script, creating a file and console handler with custom formatting.

  Args:
    log_dir (str): Directory to save the log file.
    log_filename (str): Name of the log file.

  Returns:
    logging.Logger: Configured logger instance.
  """
  os.makedirs(log_dir, exist_ok=True)
  
  logger = logging.getLogger("TrainingLogger")
  logger.setLevel(logging.DEBUG)

  # File handler for detailed logging
  file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
  file_handler.setLevel(logging.DEBUG)
  
  # Console handler for streamlined logging
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)
  
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  
  return logger


def print_and_log(message, logger_obj):
  """
  Prints a message to the console and logs it.

  Args:
    message (str): Message to print and log.
    logger_obj (logging.Logger): Logger instance to log the message.
  """
  print(message)
  logger_obj.info(message)


def get_device():
  """
  Determines the available device to run the model.

  Returns:
    str: "cuda" if a GPU is available, otherwise "cpu".
  """
  return "cuda" if torch.cuda.is_available() else "cpu"


def main(model_url):
  """
  Main function to initialize the tokenizer, model, and compute bias metrics.

  Args:
    model_url (str): Pretrained model identifier from Hugging Face.
  """

  # Sanitize model_url for use in filename
  safe_model_name = model_url.replace('/', '_').replace('\\', '_')

  # Setup the logger
  logger = setup_logger(log_filename=f"benchmark_{safe_model_name}.log")
  device = get_device()
  print_and_log(f"Using device: {device}", logger)
  
  # Load tokenizer and model from the provided model URL.
  print_and_log("Loading tokenizer and model...", logger)
  tokenizer = AutoTokenizer.from_pretrained(model_url)
  model = AutoModelForCausalLM.from_pretrained(model_url).to(device)
  print_and_log(f"Model loaded from: {model_url}", logger)
  
  # Define bias categories to evaluate.
  bias_categories = ['Caste', 'Religion', 'gender', 'disability', 'socioeconomic']
  
  print_and_log("Calculating fairness metrics...", logger)
  bias_metric_calculator = BiasMetric(model=model, tokenizer=tokenizer,
                    bias_types=bias_categories, device=device)
  
  stereotype_metric, kl_divergence_metric, bias_score_distance = bias_metric_calculator.calculate_fairness()
  
  print_and_log("Fairness metrics calculation complete.", logger)
  print_and_log("Stereotype Metric:", logger)
  print_and_log(str(stereotype_metric), logger)
  print_and_log("Distance based Bias Score:", logger)
  print_and_log(str(bias_score_distance), logger)
  print_and_log("KL Divergence based Bias Score:", logger)
  print_and_log(str(kl_divergence_metric), logger)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate bias/fairness metrics for a language model")
  parser.add_argument("--model_url", type=str,
            default="google/gemma-2-9b-it",
            help="Pretrained model identifier from Hugging Face (default: google/gemma-2-9b-it)")
  
  args = parser.parse_args()
  main(args.model_url)