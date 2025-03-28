import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from pprint import pprint
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear model using weights from a CNN.")
    parser.add_argument("--seed", type=int, required=True, help="Number of output classes for the linear model.")

    parser.add_argument("--backbone", type=str, required=True, help="Path to the checkpoint file for the model.")
    parser.add_argument("--model_path_t", type=str, required=True, help="Path to the checkpoint file for the model.")
    parser.add_argument("--model_path_tprime", type=str, required=True, help="Path to the checkpoint file for the model.")
    parser.add_argument("--t", type=int, required=True, help="Number of output classes for the linear model.")
    parser.add_argument("--tprime", type=int, required=True, help="Number of output classes for the linear model.")
    parser.add_argument("--target_layer", type=str, required=True, help="Number of output classes for the linear model.")
    parser.add_argument("--target_conv", type=str, help="Number of output classes for the linear model.")

    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    return parser.parse_args()


def extract_weights(weights, weight_path, backbone):
    if backbone == 'ResNet':
        if args.target_layer == 'block0':
            weight_name = f'blocks.{args.target_layer}.0.weight'
        else:
            weight_name = f'blocks.{args.target_layer}.{args.target_conv}.weight'
    elif backbone in ['ReLUNet', 'ReLUNetLayer9', 'ReLUNetLayer9Width256', 'ReLUNetLayer5Width256']:
        weight_name = f'blocks.{args.target_layer}.layer.weight'
    elif backbone in ['FlatViT']:
        # Load the feedforward network of Attention layer
        weight_name = f'blocks.{args.target_layer}.mlp.3.weight'
    if weight_name not in weights.keys():
        raise KeyError(f'{weight_name} not in {weights.keys()}')
    return weights[weight_name]
    # for name, param in model.named_parameters():
    #     if layer_name in name and "weight" in name:
    #         return param.data.flatten(1)  # Flatten the weight matrix for the linear model
    # raise ValueError(f"Layer {layer_name} not found in the model.") blocks.block0.0.weight'

# Step 3: Define the linear model
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.eye(self.linear.weight)
    
    def forward(self, x):
        return self.linear(x)

# Step 4: Training the model
def train_model(args):
    model_t_param = torch.load(args.model_path_t)
    model_tprime_param = torch.load(args.model_path_tprime)
    
    x = extract_weights(model_t_param, args.model_path_t, args.backbone)
    y = extract_weights(model_tprime_param, args.model_path_tprime, args.backbone)
    logger.info(f'Original weight of M_t : {x.size()}')
    logger.info(f'Original weight of M_tprime : {y.size()}')
    print(f'Original weight of M_t : {x.size()}')
    print(f'Original weight of M_tprime : {y.size()}')

    if args.backbone == 'ResNet':
        x = x.flatten(1).to(device)
        y = y.flatten(1).to(device)
    logger.info(f'Input x size : {x.size()}')
    logger.info(f'Output y size : {y.size()}')
    print(f'Input x size : {x.size()}')
    print(f'Output y size : {y.size()}')

    feature_dim = x.size(-1)

    # Define the linear model
    linear_model = LinearModel(feature_dim, feature_dim).to(device)
    # logger.info(f'Linear model\n{linear_model}')    
    print(f'Linear model\n{linear_model}')    
    
    # Configure the optimizer and loss function
    optimizer = optim.Adam(linear_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print(f'Initial MSE : {torch.mean((x-y)**2):.10f}')
    logger.info(f'Initial MSE : {torch.mean((x-y)**2):.10f}')
    
    # Train the model
    for epoch in range(args.epochs):
        # Forward pass
        outputs = linear_model(x)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.10f}")
    
    logger.info("Training complete.")

# Main function
if __name__ == "__main__":
    device = 'cuda:0'    
    args = parse_args()

    os.makedirs(f'./outputs/assume/{args.backbone}-IdentityInit-v2', exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(f'./outputs/assume/{args.backbone}-IdentityInit-v2/{args.seed}-{args.t}-{args.tprime}-{args.target_layer}.log', 'w'))
    train_model(args)
