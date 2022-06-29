import gin
import logging
import os
import wandb
from absl import app, flags

from utils import utils_params, utils_misc
from input_pipeline import datasets
from models.architectures import vgg_like
from train import Trainer
from evaluation.eval import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string('train_or_evaluate', 'evaluate',
                    "Specify train/evaluate/train_and_evaluate")
flags.DEFINE_string('data_dir', '/home/data',
                    'Specify the parent dir of IDRID_dataset, when GPU, /home/data')
flags.DEFINE_integer('n_classes', 2, '2/5-class classification problem')
flags.DEFINE_string('model_name', 'vgg_like', 'Name of the model')


def main(argv):
    # generate folder structures 'experiments'
    run_paths = utils_params.gen_run_folder()

    # set loggers, INFO and above will be logged into the file run.log
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)  # see 'experiments', path of the file run.log

    # gin-config, parse configs/config and save contents into experiments/config_operative.gin.
    # the parameters in config/gin will be used in e.g. vgg_like function
    config_file = 'configs/config.gin'
    if FLAGS.train_or_evaluate == 'evaluate':
        file_path = input("In case any change of hyperparameters after tuning, you'd better to specify the path\n"
                          "of config_operative.gin before evaluating, e.g. xxx/config_operative.gin. If you still\n"
                          "want to use the original config.gin, just press enter key to exit: \n")
        if os.path.isfile(file_path):
            config_file = file_path
            logging.info(f"Using tuned hyperparameters in {file_path}, "
                         f"instead of using hyperparameters in the original config.gin")
    gin.parse_config_files_and_bindings([config_file], [], skip_unknown=True)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    logging.info(f"------------ run folder {run_paths['path_model_id']} ------------")

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load('idrid', data_dir=FLAGS.data_dir, n_classes=FLAGS.n_classes,
                                              run_paths=run_paths)

    # model
    if FLAGS.model_name == 'vgg_like':
        model = vgg_like(input_shape=(256, 256, 3), n_classes=FLAGS.n_classes)
    else:
        raise ValueError('No such a model available.')
    model.summary(print_fn=logging.info)

    # train or evaluate
    if FLAGS.train_or_evaluate == 'train':
        logging.info("------------ Only train ------------")
        # setup wandb
        wandb.init(project='diabetic_retinopathy_train', name=run_paths['path_model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        trainer = Trainer(model, ds_train, ds_val, n_classes=FLAGS.n_classes, run_paths=run_paths)
        for _ in trainer.train():  # use _ for throwaway variables. It indicates the loop variable isn't actually used
            # continue: skip the rest of the code inside a loop for the current iteration only.
            # loop does not terminate but continues on with the next iteration.
            continue

    elif FLAGS.train_or_evaluate == 'evaluate':
        logging.info("------------ Only evaluate ------------")
        checkpoint_dir = input("Only evaluate. Muss specify the checkpoint_dir, e.g. xxx/ckpts/train :\n")
        evaluate(model, ds_test, data_dir=FLAGS.data_dir, n_classes=FLAGS.n_classes,
                 run_paths=run_paths, checkpoint_dir=checkpoint_dir)

    elif FLAGS.train_or_evaluate == 'train_and_evaluate':
        logging.info("------------ Train and evaluate ------------")
        # setup wandb
        wandb.init(project='diabetic_retinopathy_train_and_evaluate', name=run_paths['path_model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        trainer = Trainer(model, ds_train, ds_val, n_classes=FLAGS.n_classes, run_paths=run_paths)
        for _ in trainer.train():
            continue
        evaluate(model, ds_test, data_dir=FLAGS.data_dir, n_classes=FLAGS.n_classes,
                 run_paths=run_paths, checkpoint_dir=run_paths['path_ckpts_train'])

    else:
        raise ValueError("Muss specify train/evaluate/train_and_evaluate for FLAGS.train_or_evaluate")

    return 0


if __name__ == "__main__":
    app.run(main)
