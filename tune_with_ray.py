import logging
import gin
import os
import ray
from ray import tune
from ray.tune.integration.wandb import wandb_mixin

from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc


@wandb_mixin
def train_func(config, checkpoint_dir=None):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')  # ['Trainer.learning_rate=xxx', 'Trainer.total_steps=xxx', ...]
    bindings.pop()

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))  # Join all items in the list into a string, using , as separator

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    logging.info(f"------------ run folder {run_paths['path_model_id']} ------------")

    # gin-config
    # change path to absolute path of config file, e.g.:
    # GPU server: '/home/RUS_CIP/st176111/only-for-self-study-dl-lab/diabetic_retinopathy/configs/config.gin'
    # local PC: 'D:\Pycharm\PycharmProjects\only-for-self-study-dl-lab\diabetic_retinopathy\configs\config.gin'
    gin.parse_config_files_and_bindings([config_file],
                                        bindings, skip_unknown=True)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = load('idrid', n_classes=2, run_paths=run_paths)

    # model
    model = vgg_like(input_shape=(256, 256, 3), n_classes=2)

    trainer = Trainer(model, ds_train, ds_val, n_classes=2, run_paths=run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs/config.gin'))
ray.init(num_cpus=48, num_gpus=1)
analysis = tune.run(
    train_func, num_samples=50, resources_per_trial={"cpu": 48, "gpu": 1},
    config={
        "Trainer.learning_rate": tune.loguniform(1e-5, 1e-3),
        "Trainer.total_steps": tune.grid_search([2e4]),
        "vgg_like.base_filters": tune.choice([8, 16]),
        "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
        "vgg_like.dense_units": tune.choice([32, 64]),
        "vgg_like.dropout_rate": tune.uniform(0.1, 0.9),
        'wandb': {'project': 'diabetic_retinopathy_tune_with_ray'}
    }, local_dir='../ray_results')  # r'D:\Pycharm\PycharmProjects\only-for-self-study-dl-lab\ray_results'


best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max", scope='all')
best_config = best_trial.config
print(f"After hyperparameter tuning, best config: {best_config} ")
best_val_accuracy = best_trial.metric_analysis['val_accuracy']['max']
print(f"And corresponding best val_accuracy: {best_val_accuracy*100}")

# Get a dataframe for analyzing trial results.
df = analysis.dataframe(metric='val_accuracy', mode='max')
df.to_csv(os.path.join(best_trial.local_dir, 'hyperparameter_tuning_trials.csv'), index=False)
