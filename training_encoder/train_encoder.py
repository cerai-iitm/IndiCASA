from Train import train_encoder

model_names = ["all-MiniLM-L6-v2", "bert-base-cased", "ModernBERT-base"]

loss_types = ["NTXentLoss", "NTBXentLoss", "TripletLoss", "PairLoss"]

loss_params_types = ["temperature", "margin"]

training_parameters = {
    model_names[2]: {
        "loss_param": [0.1, 0.5, 1, 10, 20, 30],
        "loss_param_type": [loss_params_types[0]],
        "learning_rate": [5e-5],
        "criterion": [
            loss_types[0],
            loss_types[1]
        ],
        "max_epochs": [100],
        "early_stop_patience": [25],
        "pos_threshold": [90],
        "neg_threshold": [80],
        "device": ["cuda"],
    }
}

train_encoder(training_parameters)