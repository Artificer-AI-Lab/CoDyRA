import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import math
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import os
import pickle
import random
import logging
import loralib as lora


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

# def torch_save_lora(classifier, save_path):
#     if os.path.dirname(save_path) != "":
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     # Define the specific LoRA parameters to save
#     lora_params = ['lora_A', 'lora_B', 'lora_scaling']
    
#     # Filter the state_dict to include only the specified LoRA parameters
#     lora_state_dict = {
#         k: v for k, v in classifier.state_dict().items()
#         if any(param_name in k for param_name in lora_params)
#     }
    
#     torch.save({"state_dict": lora_state_dict}, save_path)
#     print(f"Saved parameters: {list(lora_state_dict.keys())}")
#     print("Checkpoint saved to", save_path)

def torch_save_lora(model, save_path, lora_module=None):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Define the specific LoRA parameters to save
    lora_params = ['lora_A', 'lora_B', 'lora_scaling', 'old_lora_A', 'old_lora_B',
                   'matrix', 'feature_list', 'feature_mat', 'gate', 'prototype']
    
    # Filter the state_dict to include only the specified LoRA parameters
    lora_state_dict = {
        k: v for k, v in model.state_dict().items()
        if any(param_name in k for param_name in lora_params)
    }
    
    # print(lora_state_dict.keys())
    
    # Initialize a dictionary to store LoRA ranks
    lora_ranks = {}

    # gpm_modules = [lora.GPM_LoRA_Linear, lora.Prune_GPM_LoRA_Linear, lora.InfLoRA_Linear]
    # if lora_module in gpm_modules:
    #     for name, module in model.named_modules():
    #         if isinstance(module, lora_module):  # Check for LoRA layers
    #             lora_ranks[name] = {
    #                 "rank": module.rank,
    #                 "old_rank": module.old_rank,
    #                 # 'project_type': module.project_type,
    #                 # "feature_list_shape": module.feature_list.shape,
    #                 # "feature_mat_shape": module.feature_mat.shape
    #             }
    #     # Save both the LoRA state_dict and the ranks
    #     torch.save({
    #         "state_dict": lora_state_dict,
    #         "lora_ranks": lora_ranks
    #     }, save_path)

    # else:
    if lora_module is not None:
        # Iterate over the model's named modules to capture the rank of each LoRA layer
        for name, module in model.named_modules():
            if isinstance(module, lora_module):  # Check for LoRA layers
                lora_ranks[name] = {
                    "rank": module.rank,
                    "old_rank": module.old_rank
                }
    # Save both the LoRA state_dict and the ranks
    torch.save({
        "state_dict": lora_state_dict,
        "lora_ranks": lora_ranks
    }, save_path)
    
    
    # torch.save({"state_dict": lora_state_dict}, save_path)
    # print(f"Saved parameters: {list(lora_state_dict.keys())}")
    print("Checkpoint saved to", save_path)


def torch_load_lora(model, checkpoint_path, lora_module):
    """
    Load LoRA parameters from a checkpoint and initialize the LoRA modules with saved ranks.
    
    Args:
        model (nn.Module): The model containing LoRA layers.
        checkpoint_path (str): Path to the saved checkpoint.
    
    Returns:
        model (nn.Module): The model with loaded LoRA parameters and ranks.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract LoRA state dictionary and rank information
    lora_state_dict = checkpoint["state_dict"]
    # print(lora_state_dict.keys())
    lora_ranks = checkpoint["lora_ranks"]

    # gpm_modules = [lora.GPM_LoRA_Linear, lora.Prune_GPM_LoRA_Linear, lora.InfLoRA_Linear]
    # if lora_module in gpm_modules:
    #     # Iterate over the model's named modules to reinitialize LoRA modules with the correct ranks
    #     for name, module in model.named_modules():
    #         if isinstance(module, lora_module):  # If this is a LoRA module
    #             # Check if the module's name exists in the checkpoint's saved ranks
    #             if name in lora_ranks:
    #                 saved_rank = lora_ranks[name]["rank"]
    #                 saved_old_rank = lora_ranks[name]["old_rank"]
    #                 saved_project_type = lora_ranks[name]["project_type"]
    #                 feature_list_shape = lora_ranks[name]["feature_list_shape"]
    #                 feature_mat_shape = lora_ranks[name]["feature_mat_shape"]
                    
    #                 module.register_buffer("feature_list", torch.zeros(feature_list_shape))
    #                 module.register_buffer("feature_mat", torch.zeros(feature_mat_shape))
    #                 module.project_type = saved_project_type
                    
    #                 # Re-initialize the module with the saved rank
    #                 module.rank = saved_rank
    #                 module.r = saved_rank
    #                 module.old_rank = saved_old_rank 

    #                 # Initialize lora_A, lora_B, old_lora_A, and old_lora_B based on the saved rank
    #                 module.lora_A = torch.nn.Parameter(torch.zeros((saved_rank, module.in_features)))
    #                 module.lora_B = torch.nn.Parameter(torch.zeros((module.out_features, saved_rank)))
                    
    #                 if saved_old_rank > 0:
    #                     module.old_lora_A = torch.nn.Parameter(torch.zeros((saved_old_rank, module.in_features)))
    #                     module.old_lora_B = torch.nn.Parameter(torch.zeros((module.out_features, saved_old_rank)))
    #                 else:
    #                     module.old_lora_A = None
    #                     module.old_lora_B = None    
    # else:
        # Iterate over the model's named modules to reinitialize LoRA modules with the correct ranks
    for name, module in model.named_modules():
        if isinstance(module, lora_module):  # If this is a LoRA module
            # Check if the module's name exists in the checkpoint's saved ranks
            if name in lora_ranks:
                saved_rank = lora_ranks[name]["rank"]
                saved_old_rank = lora_ranks[name]["old_rank"]
                
                # Re-initialize the module with the saved rank
                module.rank = saved_rank
                module.r = saved_rank
                module.old_rank = saved_old_rank 

                # Initialize lora_A, lora_B, old_lora_A, and old_lora_B based on the saved rank
                module.lora_A = torch.nn.Parameter(torch.zeros((saved_rank, module.in_features)))
                module.lora_B = torch.nn.Parameter(torch.zeros((module.out_features, saved_rank)))
                
                if saved_old_rank > 0:
                    module.old_lora_A = torch.nn.Parameter(torch.zeros((saved_old_rank, module.in_features)))
                    module.old_lora_B = torch.nn.Parameter(torch.zeros((module.out_features, saved_old_rank)))
                else:
                    module.old_lora_A = None
                    module.old_lora_B = None

    # Load the saved LoRA state dictionary into the model
    model.load_state_dict(lora_state_dict, strict=False)

    print(f"LoRA ranks and parameters have been successfully loaded from {checkpoint_path}.")
    
    return model

def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": classifier.state_dict()}, save_path)
    print("Checkpoint saved to", save_path)

def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    try:
        missing_keys, unexpected_keys = classifier.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
    except:
        # added for loading LoRA parameters trained with CIL
        missing_keys, unexpected_keys = classifier.load_state_dict(
            checkpoint, strict=False
        )
    # if len(missing_keys) > 0 or len(unexpected_keys) > 0:
    #     logging.info(f"Missing keys: {missing_keys}")
    #     logging.info(f"Unexpected keys: {unexpected_keys}")
    logging.info(f"Checkpoint loaded from {save_path}")
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)

    if device is not None:
        classifier = classifier.to(device)
    return classifier

def torch_load_part(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    # print(type(checkpoint["state_dict"]))
    # for k in checkpoint['state_dict'].keys():
    #     if 'router' in k:
    #         print('before',k)
    # for k in checkpoint['state_dict'].keys():
    #     if 'router' in k:
    #         checkpoint['state_dict'].pop(k)
    # for k in checkpoint['state_dict'].keys():
    #     if 'router' in k:
    #         print('after',k)
    missing_keys, unexpected_keys = classifier.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    # if len(missing_keys) > 0 or len(unexpected_keys) > 0:
    #     logging.info(f"Missing keys: {missing_keys}")
    #     logging.info(f"Unexpected keys: {unexpected_keys}")
    logging.info(f"Checkpoint loaded from {save_path}")
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)

    if device is not None:
        classifier = classifier.to(device)
    return classifier


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class kernel_layer(nn.Module):
    def __init__(self, sv, gamma):
        super(kernel_layer, self).__init__()
        self.sv = sv
        self.gamma = gamma

    def forward(self, x):
        return kernel(x, self.sv, gamma=self.gamma)


def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def one_hot_cls_acc(output, target):
    if isinstance(output, np.ndarray) and isinstance(target, np.ndarray):
        pred = np.argmax(output, axis=1)
        labels = np.argmax(target, axis=1)
        correct_predictions = np.equal(pred, labels)
        acc = np.mean(correct_predictions.astype(float)) * 100
    elif torch.is_tensor(output) and torch.is_tensor(target):
        pred = torch.argmax(output, dim=1)
        labels = torch.argmax(target, dim=1)
        correct_predictions = torch.eq(pred, labels)
        acc = torch.mean(correct_predictions.float()) * 100
    else:
        raise ValueError('Unsupported types for prediction and target.')
    return acc


def clip_classifier(classnames, template, clip_model, device="cuda"):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def encode_images(clip_model, images):
    """
        forward pass of CLIP image encoder to extract unit vector features
    """
    features = clip_model.encode_image(images)
    features /= features.norm(dim=-1, keepdim=True)


    return features

def kernel(x, X, gamma):
    """
    Args:
        x: input data
        X: static center embeddings
        gamma: Guassian kernel hyperparameter
    """
    with torch.no_grad():
        btch = 32
        ker = torch.exp(((X[:btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
        for i in range(1, math.ceil(X.size(0) / btch)):
            ker_new = torch.exp(
                ((X[i * btch:(i + 1) * btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
            ker = torch.cat((ker, ker_new), 1)
    return ker


def gaussian_kernel(x, X, gamma):
    distance = pairwise_distances(x, X, metric='euclidean', squared=True)
    return np.exp(-gamma * distance)


def linear_kernel(x, X):
    return x @ X.T


def cos_kernel(x, X):
    return 1 - linear_kernel(x, X)


def sample_per_class(dataset, n, num_classes=1000):
    indices_per_class = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset.imgs):
        indices_per_class[label].append(idx)

    sampled_indices = [idx for indices in indices_per_class for idx in np.random.choice(indices, n, replace=False)]
    return sampled_indices


class kernel_ridge_regression:
    def __init__(self, lamda=0.1, gamma=0.1):
        self.lamda = lamda
        self.gamma = gamma
        self.alpha = None
        self.kernel = None

    def train(self, X, Y):
        """
        Gaussian kernel only
        """
        self.kernel = kernel(X, X, gamma=self.gamma).cpu().numpy()
        self.alpha = np.mat(self.kernel + self.lamda * np.eye(self.kernel.shape[0])).I @ Y
        return self.alpha

    def predict(self, X, X_train):
        """
        Args:
            X: on-device tensor
            X_train: on-device tensor
        Returns:
            on-cpu numpy
        """
        predictions = kernel(X, X_train, gamma=self.gamma).cpu().numpy() @ self.alpha
        return predictions
