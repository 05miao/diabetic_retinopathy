import logging
import gin
import os
import wandb
import math

from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')  # ['Trainer.learning_rate=xxx', 'Trainer.total_steps=xxx', ...]

        # generate folder structures
        run_paths = utils_params.gen_run_folder(
            ','.join(bindings))  # Join all items in the list into a string, using , as separator

        run.name = run_paths['path_model_id'].split('/')[-1]  # used in Linux. When Windows, use '\\' instead.

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        logging.info(f"------------ run folder {run_paths['path_model_id']} ------------")

        # gin-config
        # change path to absolute path of config file, e.g.:
        # GPU server: '/home/RUS_CIP/st176111/only-for-self-study-dl-lab/diabetic_retinopathy/configs/config.gin'
        # local PC: 'D:\Pycharm\PycharmProjects\only-for-self-study-dl-lab\diabetic_retinopathy\configs\config.gin'
        gin.parse_config_files_and_bindings(
            [config_file],
            bindings, skip_unknown=True)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test = load('idrid', n_classes=2, run_paths=run_paths)

        # model
        model = vgg_like(input_shape=(256, 256, 3), n_classes=2)

        trainer = Trainer(model, ds_train, ds_val, n_classes=2, run_paths=run_paths)
        for _ in trainer.train():
            continue


config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs/config.gin'))
sweep_config = {
    'name': 'diabetic_retinopathy_tune_with_wandb_sweep',
    'method': 'bayes',
    'metric': {'name': 'best_val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'Trainer.learning_rate': {'distribution': 'log_uniform', 'min': math.log(1e-5), 'max': math.log(1e-3)},
        'Trainer.total_steps': {'values': [2e4]},
        'vgg_like.base_filters': {'values': [8, 16, 32]},
        'vgg_like.n_blocks': {'values': [2, 3, 4, 5]},
        'vgg_like.dense_units': {'values': [32, 64, 128]},
        'vgg_like.dropout_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
        'prepare.batch_size': {'values': [16, 32, 64]}
        }
}
sweep_id = wandb.sweep(sweep_config, project='diabetic_retinopathy_tune_with_wandb')
wandb.agent(sweep_id, function=train_func, count=50)
