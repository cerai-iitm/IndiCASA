import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from loss_fns import NTBXentLoss, NTXentLoss, PairLoss, TripletLoss
from encoders import ModernBertEncoder, BertEncoder, AllMiniLMEncoder
from itertools import product
from datasets import load_from_disk
import pickle
import random
import logging

GLOBAL_MAX_VAL_DIFF = 0  

def setup_logger(log_dir="logs", log_file="training.log"):
    """
    Sets up a logger to log training progress and intermediate results.

    Args:
        log_dir (str): Directory to save the log file.
        log_file (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.DEBUG)  # Log everything (DEBUG and above)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize logger
logger = setup_logger()

def print_and_log(message, logger=logger):
    """
    Print a message and log it to the logger.

    Args:
        message (str): Message to print and log.
        logger (logging.Logger): Logger instance to log the message.
    """
    print(message)
    logger.info(message)


class Trainers:
    def __init__(
        self,
        loss_param,
        loss_param_type,
        learning_rate,
        model_name,
        criterion,
        max_epochs,
        early_stop_patience,
        pos_threshold,
        neg_threshold,
        device="cuda",
    ):
        torch.manual_seed(42)
        self.loss_param = loss_param
        self.loss_param_type = loss_param_type
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.device = device
        if model_name == "all-MiniLM-L6-v2":
            self.model_class = AllMiniLMEncoder(device)
        elif model_name == "bert-base-cased":
            self.model_class = BertEncoder(device)
        elif model_name == "ModernBERT-base":
            self.model_class = ModernBertEncoder(device)
        else:
            raise NotImplementedError("Model not implemented")
        
        print_and_log(f"Model Loaded: {model_name}")

        self.load_dataset()

        print_and_log("Dataset Loaded")

        self.cont_model = ContrastiveTraining(self.model_class)
        if criterion == "NTBXentLoss":
            if loss_param_type == "temperature":
                self.criterion = NTBXentLoss(temperature=self.loss_param, device=self.device)
            else:
                raise ValueError(f"Invalid loss_param_type '{loss_param_type}' for {criterion}")
        elif criterion == "NTXentLoss":
            if loss_param_type == "temperature":
                self.criterion = NTXentLoss(temperature=self.loss_param, device=self.device)
            else:
                raise ValueError(f"Invalid loss_param_type '{loss_param_type}' for {criterion}")
        elif criterion == "PairLoss":
            if loss_param_type == "threshold":
                self.criterion = PairLoss(threshold=self.loss_param, device=self.device)
            else:
                raise ValueError(f"Invalid loss_param_type '{loss_param_type}' for {criterion}")
        elif criterion == "TripletLoss":
            if loss_param_type == "threshold":
                self.criterion = TripletLoss(threshold=self.loss_param, device=self.device)
            else:
                raise ValueError(f"Invalid loss_param_type '{loss_param_type}' for {criterion}")
        else:
            raise NotImplementedError("Criterion not implemented")
        self.optimizer = optim.AdamW(
            self.cont_model.parameters(), lr=self.learning_rate
        )

    def train(self):

        self.get_init_metrics()

        print_and_log("Training Started")

        self.enc, self.history = self.train_contrastive_model(
            model=self.cont_model,
            train_stereotypes=self.train_stereo,
            train_astereotypes=self.train_astereo,
            test_stereotypes=self.test_stereo,
            test_astereotypes=self.test_astereo,
            val_positive_pairs=self.val_pos_pairs,
            val_negative_pairs=self.val_neg_pairs,
            val_caste_positive_pairs=self.val_caste_pos_pairs,
            val_caste_negative_pairs=self.val_caste_neg_pairs,
            val_religion_positive_pairs=self.val_religion_pos_pairs,
            val_religion_negative_pairs=self.val_religion_neg_pairs,
            val_disability_positive_pairs=self.val_disability_pos_pairs,
            val_disability_negative_pairs=self.val_disability_neg_pairs,
            val_gender_positive_pairs=self.val_gender_pos_pairs,
            val_gender_negative_pairs=self.val_gender_neg_pairs,
            val_socioeconomic_positive_pairs=self.val_socioeconomic_pos_pairs,
            val_socioeconomic_negative_pairs=self.val_socioeconomic_neg_pairs,
            criterion=self.criterion,
            optimizer=self.optimizer,
            max_epochs=self.max_epochs,
            early_stop_patience=self.early_stop_patience,
            pos_threshold=self.pos_threshold,
            neg_threshold=self.neg_threshold,
        )

        print_and_log("Training Completed")

        self.history["val_pos_sim"] = [i.item() for i in self.history["val_pos_sim"]]
        self.history["val_neg_sim"] = [i.item() for i in self.history["val_neg_sim"]]
        self.history["val_pos_sim"].insert(0, self.val_possim_init.item())
        self.history["val_neg_sim"].insert(0, self.val_negsim_init.item())
        self.history["val_diff"] = [
            self.history["val_pos_sim"][i] - self.history["val_neg_sim"][i]
            for i in range(len(self.history["val_pos_sim"]))
        ]

        self.history["val_caste_pos_sim"] = [i.item() for i in self.history["val_caste_pos_sim"]]
        self.history["val_caste_neg_sim"] = [i.item() for i in self.history["val_caste_neg_sim"]]
        self.history["val_caste_pos_sim"].insert(0, self.val_possim_caste_init.item())
        self.history["val_caste_neg_sim"].insert(0, self.val_negsim_caste_init.item())
        self.history["val_caste_diff"] = [
            self.history["val_caste_pos_sim"][i] - self.history["val_caste_neg_sim"][i]
            for i in range(len(self.history["val_caste_pos_sim"]))
        ]

        self.history["val_religion_pos_sim"] = [i.item() for i in self.history["val_religion_pos_sim"]]
        self.history["val_religion_neg_sim"] = [i.item() for i in self.history["val_religion_neg_sim"]]
        self.history["val_religion_pos_sim"].insert(0, self.val_possim_religion_init.item())
        self.history["val_religion_neg_sim"].insert(0, self.val_negsim_religion_init.item())
        self.history["val_religion_diff"] = [
            self.history["val_religion_pos_sim"][i] - self.history["val_religion_neg_sim"][i]
            for i in range(len(self.history["val_religion_pos_sim"]))
        ]

        self.history["val_disability_pos_sim"] = [i.item() for i in self.history["val_disability_pos_sim"]]
        self.history["val_disability_neg_sim"] = [i.item() for i in self.history["val_disability_neg_sim"]]
        self.history["val_disability_pos_sim"].insert(0, self.val_possim_disability_init.item())
        self.history["val_disability_neg_sim"].insert(0, self.val_negsim_disability_init.item())
        self.history["val_disability_diff"] = [
            self.history["val_disability_pos_sim"][i] - self.history["val_disability_neg_sim"][i]
            for i in range(len(self.history["val_disability_pos_sim"]))
        ]

        self.history["val_gender_pos_sim"] = [i.item() for i in self.history["val_gender_pos_sim"]]
        self.history["val_gender_neg_sim"] = [i.item() for i in self.history["val_gender_neg_sim"]]
        self.history["val_gender_pos_sim"].insert(0, self.val_possim_gender_init.item())
        self.history["val_gender_neg_sim"].insert(0, self.val_negsim_gender_init.item())
        self.history["val_gender_diff"] = [
            self.history["val_gender_pos_sim"][i] - self.history["val_gender_neg_sim"][i]
            for i in range(len(self.history["val_gender_pos_sim"]))
        ]

        self.history["val_socioeconomic_pos_sim"] = [i.item() for i in self.history["val_socioeconomic_pos_sim"]]
        self.history["val_socioeconomic_neg_sim"] = [i.item() for i in self.history["val_socioeconomic_neg_sim"]]
        self.history["val_socioeconomic_pos_sim"].insert(0, self.val_possim_socioeconomic_init.item())
        self.history["val_socioeconomic_neg_sim"].insert(0, self.val_negsim_socioeconomic_init.item())
        self.history["val_socioeconomic_diff"] = [
            self.history["val_socioeconomic_pos_sim"][i] - self.history["val_socioeconomic_neg_sim"][i]
            for i in range(len(self.history["val_socioeconomic_pos_sim"]))
        ]

        with open(f"./results/history_{self.criterion}_T={self.loss_temp}_LR={self.learning_rate}.json", "w") as f:
            json.dump(self.history, f)

        return 

    def create_val_dataset(self, test_stereo, test_astereo):

        L = []
        pos_c = 0
        neg_c = 0
        for i in range(len(test_astereo)):
            d = {}
            stereo_type_pairs = test_stereo[i]
            anti_stereotype_pairs = test_astereo[i]
            pos_pairs = []
            neg_pairs = []
            for sent1 in stereo_type_pairs:
                for sent2 in stereo_type_pairs:
                    if sent1 != sent2:
                        pos_pairs.append([sent1, sent2])
                        pos_c += 1
            for sent1 in anti_stereotype_pairs:
                for sent2 in anti_stereotype_pairs:
                    if sent1 != sent2:
                        pos_pairs.append([sent1, sent2])
                        pos_c += 1
            for sent1 in stereo_type_pairs:
                for sent2 in anti_stereotype_pairs:
                    neg_pairs.append([sent1, sent2])
                    neg_c += 1
            d["pos_pairs"] = pos_pairs
            d["neg_pairs"] = neg_pairs
            L.append(d)
            del d
            gc.collect()

        val_pos_pairs = []
        val_neg_pairs = []

        for i in range(len(L)):
            val_pos_pairs.extend(L[i]["pos_pairs"])
            val_neg_pairs.extend(L[i]["neg_pairs"])

        return val_pos_pairs, val_neg_pairs


    def load_dataset(self):

        IndiCASA = load_from_disk("../IndiCASA_dataset/hf_datasets/IndiCASA")

        caste = IndiCASA["caste"]
        religion = IndiCASA["religion"]
        disability = IndiCASA["disability"]
        gender = IndiCASA["gender"]
        socioeconomic = IndiCASA["socioeconomic"]

        caste_stereo = caste["stereotypes"]
        caste_astereo = caste["anti_stereotypes"]

        train_caste_stereo, self.test_caste_stereo, train_caste_astereo, self.test_caste_astereo = (
            train_test_split(caste_stereo, caste_astereo, test_size=0.2, random_state=42)
        )

        self.val_caste_pos_pairs, self.val_caste_neg_pairs = self.create_val_dataset(
            self.test_caste_stereo, self.test_caste_astereo
        )

        religion_stereo = religion["stereotypes"]
        religion_astereo = religion["anti_stereotypes"]

        train_religion_stereo, self.test_religion_stereo, train_religion_astereo, self.test_religion_astereo = (
            train_test_split(religion_stereo, religion_astereo, test_size=0.2, random_state=42)
        )

        self.val_religion_pos_pairs, self.val_religion_neg_pairs = self.create_val_dataset(
            self.test_religion_stereo, self.test_religion_astereo
        )

        disability_stereo = disability["stereotypes"]
        disability_astereo = disability["anti_stereotypes"]

        train_disability_stereo, self.test_disability_stereo, train_disability_astereo, self.test_disability_astereo = (
            train_test_split(disability_stereo, disability_astereo, test_size=0.2, random_state=42)
        )
        
        self.val_disability_pos_pairs, self.val_disability_neg_pairs = self.create_val_dataset(
            self.test_disability_stereo, self.test_disability_astereo
        )

        gender_stereo = gender["stereotypes"]
        gender_astereo = gender["anti_stereotypes"]

        train_gender_stereo, self.test_gender_stereo, train_gender_astereo, self.test_gender_astereo = (
            train_test_split(gender_stereo, gender_astereo, test_size=0.2, random_state=42)
        )

        self.val_gender_pos_pairs, self.val_gender_neg_pairs = self.create_val_dataset(
            self.test_gender_stereo, self.test_gender_astereo
        )

        socioeconomic_stereo = socioeconomic["stereotypes"]
        socioeconomic_astereo = socioeconomic["anti_stereotypes"]

        train_socioeconomic_stereo, self.test_socioeconomic_stereo, train_socioeconomic_astereo, self.test_socioeconomic_astereo = (
            train_test_split(socioeconomic_stereo, socioeconomic_astereo, test_size=0.2, random_state=42)
        )

        self.val_socioeconomic_pos_pairs, self.val_socioeconomic_neg_pairs = self.create_val_dataset(
            self.test_socioeconomic_stereo, self.test_socioeconomic_astereo
        )

        train_stereo_comb = train_caste_stereo + train_religion_stereo + train_disability_stereo + train_gender_stereo + train_socioeconomic_stereo
        train_astereo_comb = train_caste_astereo + train_religion_astereo + train_disability_astereo + train_gender_astereo + train_socioeconomic_astereo

        train_combined = list(zip(train_stereo_comb, train_astereo_comb))

        random.seed(42)

        random.shuffle(train_combined)

        train_stereo_comb_shuff, train_astereo_comb_shuff = zip(*train_combined)

        train_stereo_comb_shuff = list(train_stereo_comb_shuff)
        train_astereo_comb_shuff = list(train_astereo_comb_shuff)

        self.train_stereo = train_stereo_comb_shuff
        self.train_astereo = train_astereo_comb_shuff

        #===============================================================

        test_stereo_comb = self.test_caste_stereo + self.test_religion_stereo + self.test_disability_stereo + self.test_gender_stereo + self.test_socioeconomic_stereo
        test_astereo_comb = self.test_caste_astereo + self.test_religion_astereo + self.test_disability_astereo + self.test_gender_astereo + self.test_socioeconomic_astereo

        test_combined = list(zip(test_stereo_comb, test_astereo_comb))

        random.seed(42)

        random.shuffle(test_combined)

        test_stereo_comb_shuff, test_astereo_comb_shuff = zip(*test_combined)

        test_stereo_comb_shuff = list(test_stereo_comb_shuff)
        test_astereo_comb_shuff = list(test_astereo_comb_shuff)

        self.test_stereo = test_stereo_comb_shuff
        self.test_astereo = test_astereo_comb_shuff

        self.val_pos_pairs, self.val_neg_pairs = self.create_val_dataset(
            self.test_stereo, self.test_astereo
        )



    def benchmark_cossim(self, val_pos_pairs, val_neg_pairs, model=None):
        if model is None:
            model = self.cont_model
        
        with torch.no_grad():
            pos_sent1 = [pair[0] for pair in val_pos_pairs]
            pos_sent2 = [pair[1] for pair in val_pos_pairs]

            pos_sent1_emb, pos_sent2_emb = model(pos_sent1, pos_sent2)
            pos_sim = torch.cosine_similarity(pos_sent1_emb, pos_sent2_emb)

            neg_sent1 = [pair[0] for pair in val_neg_pairs]
            neg_sent2 = [pair[1] for pair in val_neg_pairs]

            neg_sent1_emb, neg_sent2_emb = model(neg_sent1, neg_sent2)
            neg_sim = torch.cosine_similarity(neg_sent1_emb, neg_sent2_emb)

            val_pos_sim = pos_sim.sum() / len(pos_sim)
            val_neg_sim = neg_sim.sum() / len(neg_sim)

        return val_pos_sim, val_neg_sim

    def get_init_metrics(self):

        print_and_log("Before Training\n")

        self.val_possim_caste_init, self.val_negsim_caste_init = self.benchmark_cossim(
            self.val_caste_pos_pairs, self.val_caste_neg_pairs
        )

        self.val_possim_religion_init, self.val_negsim_religion_init = self.benchmark_cossim(
            self.val_religion_pos_pairs, self.val_religion_neg_pairs
        )

        self.val_possim_disability_init, self.val_negsim_disability_init = self.benchmark_cossim(
            self.val_disability_pos_pairs, self.val_disability_neg_pairs
        )

        self.val_possim_gender_init, self.val_negsim_gender_init = self.benchmark_cossim(
            self.val_gender_pos_pairs, self.val_gender_neg_pairs
        )

        self.val_possim_socioeconomic_init, self.val_negsim_socioeconomic_init = self.benchmark_cossim(
            self.val_socioeconomic_pos_pairs, self.val_socioeconomic_neg_pairs
        )

        self.val_possim_init, self.val_negsim_init = self.benchmark_cossim(
            self.val_pos_pairs, self.val_neg_pairs
        )

        print_and_log("Caste :")
        print_and_log(f"Val Pos Sim: {self.val_possim_caste_init:.4}")
        print_and_log(f"Val Neg Sim: {self.val_negsim_caste_init:.4f}")

        print_and_log("Religion :")
        print_and_log(f"Val Pos Sim: {self.val_possim_religion_init:.4f}")
        print_and_log(f"Val Neg Sim: {self.val_negsim_religion_init:.4f}")

        print_and_log("Disability :")
        print_and_log(f"Val Pos Sim: {self.val_possim_disability_init:.4f}")
        print_and_log(f"Val Neg Sim: {self.val_negsim_disability_init:.4f}")

        print_and_log("Gender :")
        print_and_log(f"Val Pos Sim: {self.val_possim_gender_init:.4f}")
        print_and_log(f"Val Neg Sim: {self.val_negsim_gender_init:.4f}")

        print_and_log("Socioeconomic :")
        print_and_log(f"Val Pos Sim: {self.val_possim_socioeconomic_init:.4f}")
        print_and_log(f"Val Neg Sim: {self.val_negsim_socioeconomic_init:.4f}")

        print_and_log("Combined :")
        print_and_log(f"Val Pos Sim: {self.val_possim_init:.4f}")
        print_and_log(f"Val Neg Sim: {self.val_negsim_init:.4f}")

        print_and_log("\n")


    def calc_val_loss(self, bias_type, test_stereotypes, test_astereotypes, epoch, max_epochs, model=None, criterion=None):

        val_loss = 0

        for t_stereotype, t_astereotype in tqdm(
            zip(test_stereotypes, test_astereotypes),
            desc=f"{bias_type} Validation for Epoch {epoch+1}/{max_epochs}",
        ):

            t_stereotype_emb, t_astereotype_emb = model(
                t_stereotype, t_astereotype
            )

            vloss = criterion(t_stereotype_emb, t_astereotype_emb)

            val_loss += vloss.item()

        return val_loss




    def train_contrastive_model(
        self,
        model,
        train_stereotypes,
        train_astereotypes,
        test_stereotypes,
        test_astereotypes,
        val_positive_pairs,
        val_negative_pairs,
        val_caste_positive_pairs,
        val_caste_negative_pairs,
        val_religion_positive_pairs,
        val_religion_negative_pairs,
        val_disability_positive_pairs,
        val_disability_negative_pairs,
        val_gender_positive_pairs,
        val_gender_negative_pairs,
        val_socioeconomic_positive_pairs,
        val_socioeconomic_negative_pairs,
        criterion,
        optimizer,
        max_epochs=100,
        early_stop_patience=5,
        pos_threshold=90,
        neg_threshold=80,
    ):
        """
        Train a contrastive learning model using cosine similarity thresholds with updated input parameters.

        Parameters:
            model: The neural network model.
            train_stereotypes: List of sentences for the first training set.
            train_astereotypes: List of sentences for the second training set.
            val_positive_pairs: List of tuples of two sentences representing positive pairs for validation.
            val_negative_pairs: List of tuples of two sentences representing negative pairs for validation.
            criterion: Loss function.
            optimizer: Optimizer for model training.
            max_epochs: Maximum number of training epochs.
            device: Device for computation ("cuda" or "cpu").
            early_stop_patience: Early stopping patience.
            pos_threshold: Threshold for positive cosine similarity (0-100 scale).
            neg_threshold: Threshold for negative cosine similarity (0-100 scale).

        Returns:
            model: Trained model.
            history: Training history for plotting.
        """

        best_val_loss = float("inf")
        patience_counter = 0

        # Convert thresholds to 0-1 scale
        pos_threshold = pos_threshold / 100
        neg_threshold = neg_threshold / 100

        # Initialize history dictionary for plotting
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_caste_loss": [],
            "val_religion_loss": [],
            "val_disability_loss": [],
            "val_gender_loss": [],
            "val_socioeconomic_loss": [],
            "val_pos_sim": [],
            "val_neg_sim": [],
            "val_caste_pos_sim": [],
            "val_caste_neg_sim": [],
            "val_religion_pos_sim": [],
            "val_religion_neg_sim": [],
            "val_disability_pos_sim": [],
            "val_disability_neg_sim": [],
            "val_gender_pos_sim": [],
            "val_gender_neg_sim": [],
            "val_socioeconomic_pos_sim": [],
            "val_socioeconomic_neg_sim": [],
        }

        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0

            pbar = tqdm(
                zip(train_stereotypes, train_astereotypes),
                desc=f"Epoch {epoch+1}/{max_epochs}",
            )
            for stereotype, astereotype in pbar:

                # Get embeddings
                stereotype_emb, astereotype_emb = model(stereotype, astereotype)

                # Compute loss
                loss = criterion(stereotype_emb, astereotype_emb)

                train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Validation phase
            model.eval()

            with torch.no_grad():

                caste_val_loss = self.calc_val_loss(
                    bias_type="Caste",
                    test_stereotypes=self.test_caste_stereo,
                    test_astereotypes=self.test_caste_astereo,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    model=model,
                    criterion=criterion
                )

                religion_val_loss = self.calc_val_loss(
                    bias_type="Religion",
                    test_stereotypes=self.test_religion_stereo,
                    test_astereotypes=self.test_religion_astereo,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    model=model,
                    criterion=criterion,
                )

                disability_val_loss = self.calc_val_loss(
                    bias_type="Disability",
                    test_stereotypes=self.test_disability_stereo,
                    test_astereotypes=self.test_disability_astereo,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    model=model,
                    criterion=criterion
                )

                gender_val_loss = self.calc_val_loss(
                    bias_type="Gender",
                    test_stereotypes=self.test_gender_stereo,
                    test_astereotypes=self.test_gender_astereo,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    model=model,
                    criterion=criterion,
                )
                    
                socioeconomic_val_loss = self.calc_val_loss(
                    bias_type="Socioeconomic",
                    test_stereotypes=self.test_socioeconomic_stereo,
                    test_astereotypes=self.test_socioeconomic_astereo,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    model=model,
                    criterion=criterion,
                )

                comb_val_loss = self.calc_val_loss(
                    bias_type="Combined",
                    test_stereotypes=test_stereotypes,
                    test_astereotypes=test_astereotypes,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    model=model,
                    criterion=criterion,
                )

                val_caste_posim, val_caste_negsim = self.benchmark_cossim(
                    val_caste_positive_pairs, val_caste_negative_pairs, model=model
                )

                val_religion_posim, val_religion_negsim = self.benchmark_cossim(
                    val_religion_positive_pairs, val_religion_negative_pairs, model=model
                )

                val_disability_posim, val_disability_negsim = self.benchmark_cossim(
                    val_disability_positive_pairs, val_disability_negative_pairs, model=model
                )

                val_gender_possim, val_gender_negsim = self.benchmark_cossim(
                    val_gender_positive_pairs, val_gender_negative_pairs, model=model
                )

                val_socioeconomic_possim, val_socioeconomic_negsim = self.benchmark_cossim(
                    val_socioeconomic_positive_pairs, val_socioeconomic_negative_pairs, model=model
                )

                val_possim, val_negsim = self.benchmark_cossim(
                    val_positive_pairs, val_negative_pairs, model=model
                )

            val_cossim_diff = abs(val_possim.item() - val_negsim.item())

            global GLOBAL_MAX_VAL_DIFF  # Use global tracker
            if val_cossim_diff > GLOBAL_MAX_VAL_DIFF:
                GLOBAL_MAX_VAL_DIFF = val_cossim_diff
                model_save_path = "./best_contrastive_model/best_model.pth"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                torch.save(model.state_dict(), model_save_path)  # Use torch.save instead of save_pretrained
                directory = './best_contrastive_model/'
                filename = 'Description.txt'
                file_path = os.path.join(directory, filename)
                os.makedirs(directory, exist_ok=True)
                new_content = "Loss Function : " + str(criterion) + "\n" + \
                              "Learning Rate : " + str(self.learning_rate) + "\n" + \
                              "Temperature/Threshold : " + str(self.loss_param) + "\n" + \
                              "Epoch : " + str(epoch+1)  + "\n" + \
                              "Difference in Positive & Negative Cosine Similarities : " + str(val_cossim_diff)
                with open(file_path, 'w') as file:
                    file.write(new_content)
                print_and_log(f"New best model saved with difference in positive and negative cosine similarities: {val_cossim_diff:.4f}")
                print_and_log(new_content)

            # Compute epoch metrics
            epoch_train_loss = train_loss / (
                (len(train_stereotypes) * (len(train_stereotypes) - 1)) / 2
            )

            epoch_val_loss = comb_val_loss / (
                (len(test_stereotypes) * (len(test_stereotypes) - 1)) / 2
            )

            epoch_val_caste_loss = caste_val_loss / (
                (len(self.test_caste_stereo) * (len(self.test_caste_stereo) - 1)) / 2
            )

            epoch_val_religion_loss = religion_val_loss / (
                (len(self.test_religion_stereo) * (len(self.test_religion_stereo) - 1)) / 2
            )

            epoch_val_disability_loss = disability_val_loss / (
                (len(self.test_disability_stereo) * (len(self.test_disability_stereo) - 1)) / 2
            )

            epoch_val_gender_loss = gender_val_loss / (
                (len(self.test_gender_stereo) * (len(self.test_gender_astereo) - 1)) / 2
            )

            epoch_val_socioeconomic_loss = socioeconomic_val_loss / (
                (len(self.test_socioeconomic_stereo) * (len(self.test_socioeconomic_stereo) - 1)) / 2
            )

            # Update history
            history["train_loss"].append(epoch_train_loss)

            history["val_loss"].append(epoch_val_loss)

            history["val_caste_loss"].append(epoch_val_caste_loss)
            history["val_religion_loss"].append(epoch_val_religion_loss)
            history["val_disability_loss"].append(epoch_val_disability_loss)
            history["val_gender_loss"].append(epoch_val_gender_loss)
            history["val_socioeconomic_loss"].append(epoch_val_socioeconomic_loss)

            history["val_pos_sim"].append(val_possim)
            history["val_neg_sim"].append(val_negsim)
            history["val_caste_pos_sim"].append(val_caste_posim)
            history["val_caste_neg_sim"].append(val_caste_negsim)
            history["val_religion_pos_sim"].append(val_religion_posim)
            history["val_religion_neg_sim"].append(val_religion_negsim)
            history["val_disability_pos_sim"].append(val_disability_posim)
            history["val_disability_neg_sim"].append(val_disability_negsim)
            history["val_gender_pos_sim"].append(val_gender_possim)
            history["val_gender_neg_sim"].append(val_gender_negsim)
            history["val_socioeconomic_pos_sim"].append(val_socioeconomic_possim)
            history["val_socioeconomic_neg_sim"].append(val_socioeconomic_negsim)

            # Print epoch summary

            print_and_log(f"\nEpoch {epoch+1} Summary:")

            print_and_log(f"Train Loss: {epoch_train_loss:.4f}")
            print_and_log(f"Combined Val Loss: {epoch_val_loss:.4f}\n")

            print_and_log(f"Caste Val Loss: {epoch_val_caste_loss:.4f}")
            print_and_log(f"Religion Val Loss: {epoch_val_religion_loss:.4f}")
            print_and_log(f"Disability Val Loss: {epoch_val_disability_loss:.4f}")
            print_and_log(f"Gender Val Loss: {epoch_val_gender_loss:.4f}")
            print_and_log(f"Socioeconomic Val Loss: {epoch_val_socioeconomic_loss:.4f}\n")

            print_and_log(f"Caste Val Pos Sim: {val_caste_posim:.2f}")
            print_and_log(f"Caste Val Neg Sim: {val_caste_negsim:.2f}\n")

            print_and_log(f"Religion Val Pos Sim: {val_religion_posim:.2f}")
            print_and_log(f"Religion Val Neg Sim: {val_religion_negsim:.2f}\n")

            print_and_log(f"Disability Val Pos Sim: {val_disability_posim:.2f}")
            print_and_log(f"Disability Val Neg Sim: {val_disability_negsim:.2f}\n")

            print_and_log(f"Gender Val Pos Sim: {val_gender_possim:.2f}")
            print_and_log(f"Gender Val Neg Sim: {val_gender_negsim:.2f}\n")

            print_and_log(f"Socioeconomic Val Pos Sim: {val_socioeconomic_possim:.2f}")
            print_and_log(f"Socioeconomic Val Neg Sim: {val_socioeconomic_negsim:.2f}\n")

            print_and_log(f"Combined Val Pos Sim: {val_possim:.2f}")
            print_and_log(f"Combined Val Neg Sim: {val_negsim:.2f}\n")

            # Early stopping logic
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                #torch.save(model.state_dict(), "best_contrastive_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print_and_log("Early stopping triggered!\n")                
                break

        return model, history


class ContrastiveTraining(nn.Module):

    def __init__(self, model_class):
        super(ContrastiveTraining, self).__init__()
        self.encoder = model_class

    def forward(self, stereo_texts, astereo_texts):
        stereo_embeddings = self.encoder(stereo_texts)
        astereo_embeddings = self.encoder(astereo_texts)

        return stereo_embeddings, astereo_embeddings


def train_encoder(training_parameters):

    for encoder, parameters in training_parameters.items():
        print(f"Training {encoder} model")
        param_combinations = product(
            parameters["loss_param"],
            parameters["loss_param_type"],
            parameters["learning_rate"],
            parameters["criterion"],
            parameters["max_epochs"],
            parameters["early_stop_patience"],
            parameters["pos_threshold"],
            parameters["neg_threshold"],
            parameters["device"],
        )
        for (
            loss_param,
            loss_param_type,
            learning_rate,
            criterion,
            max_epochs,
            early_stop_patience,
            pos_threshold,
            neg_threshold,
            device,
        ) in param_combinations:
            print_and_log("Training with parameters:")
            print_and_log(
                f"{loss_param_type}: {loss_param}\n Learning Rate: {learning_rate}\n Criterion: {criterion}\n Max Epochs: {max_epochs}\n Early Stop Patience: {early_stop_patience}\n Pos Threshold: {pos_threshold}\n Neg Threshold: {neg_threshold}\n Device: {device}"
            )

            trainer = Trainers(
                loss_param=loss_param,
                loss_param_type=loss_param_type,
                learning_rate=learning_rate,
                model_name=encoder,
                criterion=criterion,
                max_epochs=max_epochs,
                early_stop_patience=early_stop_patience,
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold,
                device=device,
            )
            trainer.train()

            del trainer
            gc.collect()




