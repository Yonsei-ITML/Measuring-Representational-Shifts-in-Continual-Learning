from pprint import pprint
import argparse
import task_data_loader.imagenet
from task_data_loader.split_cifar100 import cifar100_basic_transform
from utilities.configs import TrainingConfig
from utilities.evaluation import PredictionBasedEvaluator
from models import cifar10, imagenet_based_models, cifar100, mlp, cifar100_flat, imagenet_flat, flatminiresnet
from utilities.trainer import ProbeEvaluator
from utilities.utils import gpu_information_summary, set_seed
from utilities.utils import xavier_uniform_initialize
import torch
import os
import wandb

import logging
from tqdm import tqdm
from pprint import pformat
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("LOG_RUN_NAME")}_{__name__}.log', 'w'))


TASK_META_DATA = {"CUB": 200, "Scenes": 67, "ImageNet": 1000, "Flowers": 102, 'ImageNet32Task200': 5}


def main(args):
    n_gpu, _ = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)

    print("-" * 35)
    print(f"  Testing the {args.scenario} scenario.")
    print("-" * 35)

    tasks_in_scenario = args.scenario.split("2")             
    if args.scenario in ['SplitCIFAR100FixedTask50']:

        transforms = cifar100_basic_transform
        with open(f'{os.getenv("CACHE_DIR")}/{args.scenario}-{args.seed_value}.pkl', 'rb') as f:
            cl_task = pickle.load(f)
        cl_task.choose_task(args.target_task)

        training_config = TrainingConfig(
            batch_size=4096,
            num_workers=args.num_workers,
            logging_step=4000,
            early_stopping_config=None,
            nb_epochs=args.nb_epochs, #20,
            use_scheduler=True,
            nb_warmup_steps=200,
            learning_rate=0.003,
            max_steps=-1,
            prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy(), Loss()], batch_size=4096, num_workers=args.num_workers),
            is_probe=True,
            save_progress=True,
            saving_dir=args.saving_dir,
            experiment_name=args.experiment_name,
            seed_value=args.seed_value,
        )

        probe_evaluator = ProbeEvaluator(
            blocks_to_prob=[args.blocks_to_prob],
            data_stream=cl_task,
            half_precision=True,
            training_configs=training_config,
        )

        wandb_cfg = dict(
            trained_epoch=os.getenv('WANDB_TRAINED_EPOCH'),             
            **dict(vars(training_config), **vars(args)))
        if args.backbone == "FlatResNet":
            wandb_cfg['ResNetArchitecture'] = 'Flat'
        elif args.backbone == "ResNet":
            wandb_cfg['ResNetArchitecture'] = 'Curved'

        if (args.scenario == "SplitCIFAR100FixedTask50") and (args.backbone in ["FlatMiniResNetWidth64", "FlatMiniResNetWidth32", "FlatMiniResNetWidth16", "FlatMiniResNetWidth8"]):
            internal_channel = int(args.backbone.split('Width')[-1])
            wandb_cfg['internal_channel'] = internal_channel
            model = flatminiresnet.FlatMiniResNet(task_num=50, internal_channel=internal_channel)

        model.load_state_dict(torch.load(args.model_path))

        run = wandb.init(project=os.getenv("WANDB_PROJECT"), config=wandb_cfg)
        wandb.run.name= f"Probe_{os.getenv('RUN_NAME_PREFIX')}_{training_config.strategy}_Target{args.target_task}_block{args.blocks_to_prob}_{type(cl_task).__name__}_{type(model).__name__}_{training_config.seed_value}"
        wandb.log({'Trained Tasks Num': args.trained_task_id})
        probe_results = probe_evaluator.probe(model=model, probe_caller=args.probe_caller)
        logger.info(probe_results)                 
    elif args.scenario in ['ImageNet100Resized32FixedTask50']:
        with open(f'{os.getenv("CACHE_DIR")}/{args.scenario}-{args.seed_value}.pkl', 'rb') as f:
            cl_task = pickle.load(f)
        cl_task.choose_task(args.target_task)

        training_config = TrainingConfig(
            batch_size=256,
            num_workers=args.num_workers,
            logging_step=4000,
            early_stopping_config=None,
            nb_epochs=args.nb_epochs,
            use_scheduler=True,
            nb_warmup_steps=200,
            learning_rate=args.lr,
            max_steps=-1,
            prediction_evaluator=PredictionBasedEvaluator(
                metrics=[Accuracy(), Loss()], batch_size=4096, num_workers=30
            ),
            is_probe=True,
            save_progress=True,
            saving_dir=args.saving_dir,
            experiment_name=args.experiment_name,
            seed_value=args.seed_value,
        )

        probe_evaluator = ProbeEvaluator(
            blocks_to_prob=[args.blocks_to_prob], 
            data_stream=cl_task,
            half_precision=True,
            training_configs=training_config,
        )
        if args.backbone == "MiniResNet":
            if args.supcon:
                model = imagenet_based_models.MiniResNetSupCon(
                    back_bone_path=args.model_path, prediction_layers=prediction_layer, is_probe=args.is_probe,
                )
            else:
                model = imagenet_based_models.MiniResNet(
                    back_bone_path=args.model_path, prediction_layers=prediction_layer, is_probe=args.is_probe,
                )
        elif args.backbone == "FlatMiniResNet":
            if args.supcon:
                raise KeyError(f'Cannot support')
            else:
                model = imagenet_flat.FlatMiniResNet(task_num=int(args.scenario.split('Task')[-1]))

        logger.info(f'model : {pformat(list(model.state_dict()))}')
        model.load_state_dict(torch.load(args.model_path))
        run = wandb.init(project=os.getenv("WANDB_PROJECT"), config=dict(vars(training_config), **vars(args)))
        wandb.run.name= f"Probe_{training_config.strategy}_{type(cl_task).__name__}_{type(model).__name__}_{training_config.seed_value}"
        wandb.log({'Trained Tasks Num': args.trained_task_id})
        probe_results = probe_evaluator.probe(model=model, probe_caller=args.probe_caller)
        logger.info(probe_results)            
    else:
        raise Exception("Scenario is not supported!", args.scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saving_dir",
        help="Path of the model the CL Scenario will continue or the probing will happen on.",
        type=str,
    )

    parser.add_argument(
        "--is_probe",
        help="Probes a model rather than training a CL scenario",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--blocks_to_prob",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_path",
        help="Path of the model the CL Scenario will continue or the probing will happen on.",
        type=str,
    )
    parser.add_argument(
        "--scenario",
        help="Name of the CL scenario, indicative of task sequence.",
        type=str,
    )
    parser.add_argument(
        "--probe_caller",
        help="Indicates which task is calling the prob.",
        type=str,
    )

    parser.add_argument(
        "--data_root",
        help="Path to the root data.",
        type=str,
    )

    parser.add_argument(
        "--strategy",
        help="Name of the CL strategy, indicative of forgetting mitigation strategy",
        type=str,
        choices=["FineTuning", "LwF", "EWC8000", "EWC500"],
        default="FineTuning",
    )

    parser.add_argument(
        "--probing_train_data",
        help="Indicates the training dataset for the linear prob",
        type=str,
        choices=["ImageNet", "Scenes"],
        default="ImageNet",
    )

    parser.add_argument(
        "--target_task",
        help="Indicates the training dataset for the linear prob",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--trained_task_id",
        help="Indicates the trained dataset for loaded model",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--backbone",
        help="Either VGG or ResNet backbone.",
        type=str,
        default="ResNet",
    )

    parser.add_argument("--experiment_name", help="Optional name for the experiment.", type=str, default=None)
    parser.add_argument(
        "--use_different_seed",
        help="Indicates if we want to use different seeds for different models.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--nb_epochs",
        help="The number of epochs to be trained.",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--num_workers",
        help="The number of workers for data loader",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--lr",
        help="learning rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed_value",
        help="Random seed value for this experiment.",
        type=int,
        default=3407,
    )

    parser.add_argument(
        "--supcon",
        help="Whether to use supcon or not.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--width",
        help="Width for ReLUNet",
        type=int
    )

    parser.add_argument(
        "--depth",
        help="depth of ReLUNet",
        type=int
    )

    parser.add_argument(
        "--train_dataset_size",
        help="depth of ReLUNet",
        default=2000,
        type=int
    )
    parser.add_argument(
        "--test_dataset_size",
        help="depth of ReLUNet",
        default=100,
        type=int
    )
    parser.add_argument(
        "--seq_len",
        help="depth of ReLUNet",
        default=5,
        type=int
    )


    args = parser.parse_args()

    main(args=args)
