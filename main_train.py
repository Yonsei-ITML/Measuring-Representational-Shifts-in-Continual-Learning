from pprint import pprint
import argparse
import task_data_loader.imagenet
from models.imagenet_based_models import PredictionLayerConfig
from task_data_loader.split_cifar10 import cifar10_basic_transform
from task_data_loader.split_cifar100 import cifar100_basic_transform
from task_data_loader.scenarios import (
    SplitCIFAR100FixedTask50,
    ImageNet100Resized32FixedTask50,
)
from utilities.configs import TrainingConfig
from utilities.evaluation import RepresentationBasedEvaluator, PredictionBasedEvaluator
from utilities.metrics import CKA, L2, Accuracy, Loss
from models import cifar10, imagenet_based_models, cifar100, mlp, cifar100_flat, imagenet_flat, imagenet_funnel, flatminiresnet, flatvit
from utilities.trainer import ModelCoach, ProbeEvaluator
from utilities.utils import gpu_information_summary, set_seed, EarlyStoppingConfig
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


TASK_META_DATA = {
    "CUB": 200, "Scenes": 67, "ImageNet": 1000, "Flowers": 102, 
    'ImageNet32Task200': 5, 'ImageNet32Task500': 2, 
    'ImageNet100Resized32Task20':5, 'ImageNet100Resized32Task50': 5,
    'ImageNet100Resized32FixedTask50' : 5,
    'ImageNet100Resized32FixedTask200': 5,
    }


def main(args):
    n_gpu, _ = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)

    print("-" * 35)
    print(f"  Testing the {args.scenario} scenario.")
    print("-" * 35)

    if args.scenario == "SplitCIFAR10":
        cl_task = SplitCIFAR10(root=args.data_root, transforms=cifar10_basic_transform)
    elif args.scenario == "SplitCIFAR100FixedTask50":
        cl_task = SplitCIFAR100FixedTask50(root=args.data_root, transforms=cifar100_basic_transform)
    elif args.scenario.startswith("ImageNet"):
        if args.backbone == "MiniResNet":
            transforms = [
                task_data_loader.imagenet.mini_train_transform,
                task_data_loader.imagenet.mini_valid_transform,
            ]
        elif args.backbone == "FlatMiniResNet":
            transforms = [
                task_data_loader.imagenet.mini_flat_train_transform,
                task_data_loader.imagenet.mini_flat_valid_transform,
            ]            
        else:
            transforms = [task_data_loader.imagenet.train_transform, task_data_loader.imagenet.valid_transform]

        if args.scenario == 'ImageNet100Resized32FixedTask50':
            cl_task = ImageNet100Resized32FixedTask50(root=args.data_root, transforms=transforms)
        else:
            raise Exception("Scenario is not supported!", args.scenario)
    else:
        raise Exception("Scenario is not supported!", args.scenario)
        
    with open(f'{os.getenv("CACHE_DIR")}/local_datasets/{args.scenario}-{args.seed_value}.pkl', 'wb') as f:
        pickle.dump(cl_task, f)
    # load the proper model
    if args.scenario.startswith("ImageNet"):
        
        if args.scenario == 'ImageNet100Resized32FixedTask50':
            prediction_layer = [
                PredictionLayerConfig(task_id=f'Task_{task_id}', nb_classes=TASK_META_DATA[args.scenario])
                for task_id in range(50)
            ]  
        else:
            tasks_in_scenario = args.scenario.split("2")
            
            prediction_layer = [
                PredictionLayerConfig(task_id=task.lower(), nb_classes=TASK_META_DATA[task])
                for task in tasks_in_scenario[1:]
            ]
        
        if args.backbone == "FlatMiniResNet":
            if args.supcon:
                raise KeyError(f'Cannot support')
            else:
                model = imagenet_flat.FlatMiniResNet(task_num=int(args.scenario.split('Task')[-1]))
        else:
            raise Exception("The backbone model is not supported for CIFAR10 data. --backbone: ", args.backbone)
    else:
        if (args.backbone == "FlatMiniResNet") and (args.scenario == "SplitCIFAR100FixedTask50"):
            model = flatminiresnet.FlatMiniResNet(task_num=50, internal_channel=args.internal_channel)
        elif (args.backbone == "FlatViT") and (args.scenario == "SplitCIFAR100FixedTask50"):
            model = flatvit.FlatVisionTransformer(task_num=50, internal_channel=args.internal_channel)
        else:
            raise Exception("The backbone model is not supported for CIFAR10 data. --backbone: ", args.backbone)

        model.apply(xavier_uniform_initialize)

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logging_step=1000,
        early_stopping_config=None,
        nb_epochs=args.nb_epochs,
        use_scheduler=True,
        nb_warmup_steps=args.nb_warmup_steps,
        learning_rate=args.learning_rate,
        max_steps=-1,
        prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy(), Loss()], batch_size=512, num_workers=8),
        representation_evaluator=None,
        is_probe=False,
        save_progress=True,
        saving_dir=args.saving_dir,
        strategy=args.strategy,
        experiment_name=args.experiment_name,
        use_different_seed=args.use_different_seed,
        seed_value=args.seed_value,
        use_sup_con=args.supcon,
        nb_epochs_supcon=100,
    )
    run = wandb.init(project=os.getenv("WANDB_PROJECT"), config=dict(vars(training_config), **vars(args)))
    wandb.run.name= f"{training_config.strategy}_{type(cl_task).__name__}_{type(model).__name__}_{training_config.seed_value}"
    trainer = ModelCoach(model=model, data_stream=cl_task, config=training_config)
    training_results_last_iter = trainer.train()
    pprint(training_results_last_iter)


if __name__ == "__main__":
    # NOTE : for torch version < 2.0, it does not work
    # torch.set_default_device('cuda')
    # torch.multiprocessing.set_start_method('spawn')

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
        "--backbone",
        help="Either VGG or ResNet backbone.",
        type=str,
        choices=["MiniResNet", 'ReLUNet', 'FlatResNet', 'FlatMiniResNet', 'FlatViT'],
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
        "--batch_size",
        help="The number of workers for data loader",
        type=int,
        default=256,
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
        "--internal_channel",
        help="Internal Channel for FlatMiniResNet",
        type=int
    )

    parser.add_argument(
        "--input_vector_dim",
        help="input_vector_dim for ReLUNet",
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
    
    parser.add_argument(
        "--nb_warmup_steps",
        help="depth of ReLUNet",
        default=200,
        type=int
    )    
    parser.add_argument(
        "--learning_rate",
        help="depth of ReLUNet",
        default=1e-5,
        type=float
    )       
    args = parser.parse_args()

    main(args=args)
