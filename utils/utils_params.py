import os
import datetime


def gen_run_folder(path_model_id=''):  # see the structure of the directory 'experiments'
    run_paths = dict()

    if not os.path.isdir(path_model_id):  # True
        # how to get path_model_root, for example,
        # __file__: 'xxx\diabetic_retinopathy\utils\utils_params.py'
        # os.path.dirname(__file__): current dir 'xxx\diabetic_retinopathy\utils'
        # os.pardir: '..'
        # os.path.join: 'xxx\diabetic_retinopathy\utils\..\..\experiments'
        # os.path.abspath: 'xxx\experiments'
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
        date_creation = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
        run_id = 'run_' + date_creation
        if path_model_id:  # When train, False; when tune, True
            run_id += '_' + path_model_id
        # run_paths = {'path_model_id': 'xxx\experiments\run_...'}
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        run_paths['path_model_id'] = path_model_id

    # see the structure of the directory 'experiments'
    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'train')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')
    # summary for Tensorflowboard
    run_paths['path_summary_tensorboard'] = os.path.join(run_paths['path_model_id'], 'summary_tensorboard')

    # Create folders
    for k, v in run_paths.items():  # dictionary's (key, value) tuple pairs e.g. ('path_model_id','xxx\experiments\...')
        if any([x in k for x in ['path_model', 'path_ckpts']]):
            # k='path_model_id': x='path_model' is in 'path_model_id', x='path_ckpts' not -> any([True,False]) is True
            # k='path_ckpts_train': 'path_model' not in k, 'path_ckpts' yes -> any([False,True]) is True
            if not os.path.exists(v):  # 'xxx\experiments\run_...' or '...run_...\ckpts' doesn't exist
                os.makedirs(v, exist_ok=True)  # create directories

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):  # 'path_logs_train' is in 'path_logs'
            if not os.path.exists(v):  # 'xxx\experiments\run_...\logs\run.log' doesn't exist
                os.makedirs(os.path.dirname(v), exist_ok=True)  # create directory 'xxx\experiments\run_...\logs'
                with open(v, 'a'):  # open the file 'run.log' for writing, appending to the end of file if it exists
                    pass  # atm file creation is sufficient

    return run_paths  # dict


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


# https://github.com/google/gin-config/issues/154
def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data["/".join([name, k])] = v

    return data
