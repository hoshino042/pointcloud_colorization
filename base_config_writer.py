import configparser
config = configparser.ConfigParser()
config["hyperparameters"] = {
    "batch_size": 8,
    "epoch": 200,
    "category": "chair",
    "num_pts": 4096,
    "lr_g": 0.001,
    "lr_d": 0.0001
}

with open("base_config.ini", "w") as configfile:
    config.write(configfile)