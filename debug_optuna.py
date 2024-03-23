import optuna 

study = optuna.load_study(storage= "sqlite:///results/finetuning/recording_ultra_light_reb.db", study_name = "sulfur_QR_0_0")
print(study.trials[0].state == optuna.trial.TrialState.FAIL)

print(study.best_params)