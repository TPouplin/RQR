from source.parameters import best_parameter_dict
from source.model import Q_model, SQ_model, SWS_model
from data.dataset import GetDataset, Dataset


from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint



# default_parameters = {
#     "dropout":0.1,
#     "epochs":1000,
#     "lr":0.0005,
#     "penalty":0,
#     "batch_size":64
# }

def run_experiment(config, data=None):
    # if not config["finetuning"]:
    #     best_parameters = best_parameter_dict[config["dataset_name"]][config["loss"]]
        
    #     for key in default_parameters:
    #         if (key not in config) or (config[key] is None) :
    #             if key in best_parameters:
    #                 config[key] = best_parameters[key]
    #             else:
    #                 config[key] = default_parameters[key]
                
    L.pytorch.seed_everything(config["random_seed"], workers=True)
    if data is None:
        X, y = GetDataset(config["dataset_name"], "data")
    else:
        X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_ratio"], random_state=config["random_seed"])
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config["val_ratio"], random_state=config["random_seed"])
    
    
    # zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train)
    X_train = scalerX.transform(X_train)
    X_val = scalerX.transform(X_val)
    X_test = scalerX.transform(X_test)
    
    # scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train))
    y_train = y_train / mean_ytrain
    y_val = y_val / mean_ytrain
    y_test = y_test / mean_ytrain
    
    train_dataset = Dataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = Dataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = Dataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # print("Data loaded")
    
    quantiles = [(1-config["coverage"])/2, 1-(1-config["coverage"])/2]
    
    if config["loss"] == "SQR":
        model = SQ_model(coverage=config["coverage"], x_shape=X_train.shape[1], hidden_size=64, dropout=config["dropout"], lr=config["lr"], loss = config["loss"], penalty=config["penalty"])
        model.narrowest = False
    elif config["loss"] == "SWS":
        model = SWS_model(coverage=config["coverage"], x_shape=X_train.shape[1], hidden_size=64, dropout=config["dropout"], lr=config["lr"], loss = config["loss"], penalty=config["penalty"])
    elif config["loss"] == "SQRN":
        model = SQ_model(coverage=config["coverage"], x_shape=X_train.shape[1], hidden_size=64, dropout=config["dropout"], lr=config["lr"], loss = config["loss"], penalty=config["penalty"])
        model.narrowest = True
    else:
        model = Q_model(coverage=config["coverage"], x_shape=X_train.shape[1], hidden_size=64, dropout=config["dropout"], lr=config["lr"], loss = config["loss"], penalty=config["penalty"])
    
    date = datetime.now().strftime("%Y%m%d%H%M%S")
        
    logger = TensorBoardLogger(save_dir="results/logs/", name=f'''{config["loss"]}_{config["dataset_name"]}_{config["random_seed"]}_{date}''')
    
    # print("config : ", config)
    
    logger.log_hyperparams(config)
    
    checkpoint_callback = ModelCheckpoint(dirpath=f'''results/logs/{config["loss"]}_{config["dataset_name"]}_{config["random_seed"]}_{date}''', save_top_k=1, monitor="val_loss", filename="best_checkpoint")
    
    if config["finetuning"]:
        trainer = L.Trainer(max_epochs=config["epochs"],accelerator="gpu", logger=False, deterministic=True,callbacks=[checkpoint_callback], enable_progress_bar=False, enable_model_summary=False)
    else:
        trainer = L.Trainer(max_epochs=config["epochs"],accelerator="gpu", deterministic=True, logger=logger, callbacks=[checkpoint_callback], enable_progress_bar=True)

    trainer.fit(model, train_loader, val_loader)
            
    results = trainer.test(model, test_loader, ckpt_path= checkpoint_callback.best_model_path)
    
    del train_dataset
    del val_dataset
    del test_dataset
    del train_loader
    del val_loader
    del test_loader
    del model 
    del trainer
    torch.cuda.empty_cache()

    return checkpoint_callback.best_model_score, results[0]