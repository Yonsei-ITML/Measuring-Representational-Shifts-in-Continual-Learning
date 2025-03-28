import os
import time
import wandb
import torch
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint, pformat

from sklearn.linear_model import LinearRegression

import task_data_loader.imagenet
from utilities.utils import gpu_information_summary, set_seed
from models import cifar10, cifar100, mlp, cifar100_flat, imagenet_flat, flatminiresnet


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("LOG_RUN_NAME")}_{__name__}.log', 'w'))


def compute_width(_model, _target_layer):
    block = _model.blocks[_target_layer]
    if isinstance(block, torch.nn.Sequential):
        layer = block[0] 
    elif isinstance(block, cifar100.ResidualBlock):
        layer = block.conv2
    elif isinstance(block, cifar100_flat.ResidualBlock):
        layer = block.conv2
    elif isinstance(block, imagenet_flat.ResidualBlock):
        layer = block.conv2
    elif isinstance(block, flatminiresnet.ResidualBlock):
        layer = block.conv2
    else:
        raise TypeError()
    return torch.norm(layer.weight, p='fro').item()


def compute_feature(_model, _model_task_index, _dataset, _target_layer_index, seed):
    fname = f'{os.getenv({"CACHE_DIR"})}/{args.scenario}-{seed}-{args.backbone}-task{_model_task_index}-layer{_target_layer_index}.pkl'
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            outputs = pickle.load(f)        
    else:
        outputs = []
        for data in tqdm(_dataset, desc=f'Computing feature of M_{_model_task_index}^{_target_layer_index}...'):
            features, targets = data
            features = features.unsqueeze(0)            
            
            if isinstance(_model, mlp.ReLUNetComplexSymbolCounting):
                features = _model.embedding(features)
            
            for block_name, operations in _model.blocks.items():

                features = operations(features)
                if block_name == f'{_target_layer_index}':
                    break

            outputs.append(torch.flatten(features, 1).squeeze().detach().cpu().numpy())
        outputs = np.array(outputs)
        with open(fname, 'wb') as f:
            pickle.dump(outputs, f)

    return outputs

def compute_eta(_feature_t):
    feature_dim = _feature_t.shape[-1]
    outputs = np.linalg.norm(_feature_t, ord=2, axis=-1)
    min_val = min(outputs)
    max_val = max(outputs)

    return min_val, max_val, min_val / np.sqrt(feature_dim), max_val / np.sqrt(feature_dim)

def compute_sigma(_feature_t, _feature_tprime):
    outputs = np.linalg.norm(_feature_t-_feature_tprime, ord=2, axis=-1)
    return max(outputs)


def compute_eps(_feature_t, _feature_tprime):
    reg = LinearRegression(fit_intercept=False).fit(_feature_tprime, _feature_t)

    _feature_tprime_transformed = reg.predict(_feature_tprime)

    eps = np.linalg.norm(_feature_t- _feature_tprime_transformed, ord=2, axis=-1)
    eps = np.max(eps)
    return eps


def main(args):
    n_gpu, _ = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)

    print("-" * 35)
    print(f"  Testing the {args.scenario} scenario.")
    print("-" * 35)

   if args.scenario == 'SplitCIFAR100FixedTask50':
        with open(f'{os.getenv("CACHE_DIR")}/{args.scenario}-{args.seed_value}.pkl', 'rb') as f:
            cl_task = pickle.load(f)
        dataset_t = cl_task.tasks[args.t].train
        logger.info(f'dataset_t : {dataset_t}')
        
        if args.backbone == 'FlatResNet':
            model_t = cifar100_flat.FlatResNetCIFAR100(task_num=50)
            loaded_model_t = torch.load(args.model_path_t)

            logger.info(f'model_t.feature_size : {model_t.feature_size}')
            logger.info(f'Architecture : \n{model_t}')

            logger.info(f'parameter ')
            for name, tensor in loaded_model_t.items():
                logger.info(f'{name}:{tensor.size()}')
            model_t.load_state_dict(loaded_model_t)     
            logger.info(f'model_t : \n{model_t}')  

            model_tprime = cifar100_flat.FlatResNetCIFAR100(task_num=50)
            model_tprime.load_state_dict(torch.load(args.model_path_tprime))       
            logger.info(f'model_tprime : \n{model_t}')  
        elif args.backbone in ['FlatMiniResNetWidth32', 'FlatMiniResNetWidth16', 'FlatMiniResNetWidth8']:
            internal_channel = int(args.backbone.split('Width')[-1])
            task_num = int(args.scenario.split('Task')[-1])
            model_t = flatminiresnet.FlatMiniResNet(task_num=task_num, internal_channel=internal_channel)
            model_t.load_state_dict(torch.load(args.model_path_t))     

            model_tprime = flatminiresnet.FlatMiniResNet(task_num=task_num, internal_channel=internal_channel)
            model_tprime.load_state_dict(torch.load(args.model_path_tprime))       
            logger.info(f'model_tprime : \n{model_t}')  

        feature_t = compute_feature(model_t, args.t, dataset_t, f'block{args.k}', args.seed_value)
        feature_tprime = compute_feature(model_tprime, args.tprime, dataset_t, f'block{args.k}', args.seed_value)

        width_t = compute_width(model_t, f'block{args.k}')
        width_tprime = compute_width(model_tprime, f'block{args.k}')
        eta_min, eta_max, eta_min_normalized, eta_max_normalized = compute_eta(feature_t)

        sigma = compute_sigma(feature_t, feature_tprime)
        
        eps = compute_eps(feature_t, feature_tprime)

        with open(f'./outputs/extended-measurements-v2-{args.scenario}-{args.backbone}.csv', 'a') as f:
            logs = [
                args.seed_value,
                args.k, args.t, args.tprime,
                eta_min, eta_max, eta_min_normalized, eta_max_normalized,
                sigma, eps,
                width_t, width_tprime,
            ]
            logs = list(map(str, logs))
            f.write(','.join(logs)+ '\n')   

    elif args.scenario in ['ImageNet100Resized32FixedTask50']:
        with open(f'{os.getenv("CACHE_DIR")}/{args.scenario}-{args.seed_value}.pkl', 'rb') as f:
            cl_task = pickle.load(f)        

        dataset_t = cl_task.tasks[args.t].train
        logger.info(f'dataset_t : {dataset_t}')

        if args.backbone == 'FlatMiniResNet':
            model_t = imagenet_flat.FlatMiniResNet(task_num=int(args.scenario.split('Task')[-1]))
            model_t.load_state_dict(torch.load(args.model_path_t))     
            logger.info(f'model_t : \n{model_t}')  
            
            model_tprime = imagenet_flat.FlatMiniResNet(task_num=int(args.scenario.split('Task')[-1]))
            model_tprime.load_state_dict(torch.load(args.model_path_tprime))       
            logger.info(f'model_tprime : \n{model_t}')  

        feature_t = compute_feature(model_t, args.t, dataset_t, f'block{args.k}', args.seed_value)
        feature_tprime = compute_feature(model_tprime, args.tprime, dataset_t, f'block{args.k}', args.seed_value)

        width_t = compute_width(model_t, f'block{args.k}')
        width_tprime = compute_width(model_tprime, f'block{args.k}')
        eta_min, eta_max, eta_min_normalized, eta_max_normalized = compute_eta(feature_t)

        sigma = compute_sigma(feature_t, feature_tprime)
        
        eps = compute_eps(feature_t, feature_tprime)

        with open(f'./outputs/extended-measurements-v2-{args.scenario}-{args.backbone}.csv', 'a') as f:
            logs = [
                args.seed_value,
                args.k, args.t, args.tprime,
                eta_min, eta_max, eta_min_normalized, eta_max_normalized,
                sigma, eps,
                width_t, width_tprime,
            ]
            logs = list(map(str, logs))
            f.write(','.join(logs)+ '\n')             
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
        "--data_root",
        help="Path to the root data.",
        type=str,
    )

    parser.add_argument(
        "--scenario",
        help="Name of the CL scenario, indicative of task sequence.",
        type=str,
    )

    parser.add_argument(
        "--k",
        help="blocks_to_prob",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--t",
        help="target trained task",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--tprime",
        help="target trained task",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model_path_tprime",
        help="`t'`; Target task",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_path_t",
        help="`t`; Trained dataset index",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--backbone",
        help="Either VGG or ResNet backbone.",
        type=str,
        default="ResNet",
    )

    parser.add_argument(
        "--seed_value",
        default=123,
        type=int,
    )
    args = parser.parse_args()

    main(args=args)
