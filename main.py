import torch
from tqdm import tqdm
import random
import numpy as np
import os
import yaml
import argparse
from scenario_datasets import build_dataset
from scenario_datasets.utils import build_data_loader
from scenario_datasets.collections import CIFAR100, MNIST
from utils import *
from models.factory import get_model_loader, get_trainer

LORA_MAPPING = {
    "q": ['q_proj'],
    "k": ['k_proj'],
    "v": ['v_proj'],
    "kv": ['k_proj', 'v_proj'],
    'o': ['out_proj'],
    'qkv': ['q_proj', 'k_proj', 'v_proj'],
    'qkvo': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    'in': ['ffn_in'],
    'out': ['ffn_out'],
    'inout': ['ffn_in', 'ffn_out'],
    'kvinout': ['k_proj', 'v_proj', 'ffn_in', 'ffn_out'],
    'qkvinout': ['q_proj', 'k_proj', 'v_proj', 'ffn_in', 'ffn_out'],
    'qkvoinout': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'ffn_in', 'ffn_out'],
}

###########################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Training configuration")

    # General arguments
    parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducibility")
    parser.add_argument('--datasets', type=list, default=["aircraft", "caltech101", "dtd", "eurosat", "oxford_flowers",
                                                          "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="List of dataset names")
    parser.add_argument('--num_shots', type=int, default=16, help="Number of shots for few-shot learning")
    parser.add_argument('--data_dir', type=str, default='datasets', help="Path to the dataset directory")

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--batch_size_eval', type=int, default=128, help="Batch size for evaluation")
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=0)

    # Evaluation settings
    parser.add_argument('--augmentation_time', type=int, default=1, help="Number of augmentation passes")
    
    # method
    parser.add_argument("--method", type=str, default="codyra")
    
    # lora setting
    parser.add_argument('--backbone_type', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained_weight', type=str, default='openai')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--target_encoder', type=str, default='vision')
    parser.add_argument('--target_modules_abbrev', type=str, default='qkvo')
    parser.add_argument('--target_modules', type=list, default=None)
    parser.add_argument("--max_kappa", type=float, default=7e-4, help="Maximum lambda for regularization")
    parser.add_argument("--lambda_num", type=int, default=50, help="Number of lambda steps")
    parser.add_argument('--dense_iters', type=float, default=0.5, help="Number of dense iterations")
    
    
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)

    return parser.parse_args()

def load_config_from_yaml(args):
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args

def setup_logging(logfilename):
    import sys
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)d] - %(message)s")

    # Create file handler
    file_handler = logging.FileHandler(logfilename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main(cfg):
    
    log_dir = "logs/{}/{}_{}/{}/{}/rank_{}/".format(
        args.method,
        args.backbone_type,
        args.pretrained_weight,
        args.target_encoder,
        args.target_modules_abbrev,
        args.rank)
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, "train.log")

    # Set up logging
    setup_logging(logfilename)
    
    
    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_sequence = cfg.datasets
    logging.info(f"Multi-task dataset sequence: {dataset_sequence}")

    # Results
    last_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))

    cfg.previous_class_num = 0
    current_class_names = []
    cfg.seen_classes = []

    """
    Loading model
    """
    logging.info('Loading pretrained CLIP model...')
    model_loader = get_model_loader(cfg)   
    model, train_preprocess, val_preprocess, tokenizer = model_loader(vars(cfg)) 
        
    """
    Training on dataset sequence
    """
    for task_id, train_dataset in enumerate(dataset_sequence):
        logging.info(f"------------------ Start training on task-{task_id + 1}: dataset-{train_dataset}. ---------------------")
        if train_dataset == "cifar100":
            dataset = CIFAR100(num_shots=cfg.num_shots, preprocess=train_preprocess, val_transform=val_preprocess,
                            batch_size=cfg.batch_size, location=cfg.data_dir)
        elif train_dataset == "mnist":
            dataset = MNIST(num_shots=cfg.num_shots, preprocess=train_preprocess, val_transform=val_preprocess,
                            batch_size=cfg.batch_size, location=cfg.data_dir)
        else:
            dataset = build_dataset(train_dataset, cfg.data_dir, cfg.num_shots, val_preprocess)

        current_class_names += dataset.classnames
        cfg.increment = len(dataset.classnames)
        cfg.current_class_num = len(current_class_names)

        if train_dataset == "cifar100" or train_dataset == "mnist":
            train_loader = dataset.train_loader
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg.batch_size, tfm=train_preprocess,
                                            is_train=True, shuffle=True, augmentation_time=cfg.augmentation_time)
        
        trainer = get_trainer(cfg)
        dataset_name = train_dataset
        trainer(cfg, model, tokenizer, dataset_name, dataset, train_loader)         
        cfg.trained_class_num = cfg.current_class_num
        

        # """
        # Testing stage: test on every dataset (both trained & untrained) after training on each dataset
        # """
        evaluation(cfg, dataset_sequence, task_id, val_preprocess, last_acc_table, model, tokenizer)
                    
    upper_triangle_no_diag = np.triu(last_acc_table, k=1)
    masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
    transfer_acc = np.mean(masked_matrix, axis=0)
    transfer_avg_acc = np.mean(transfer_acc)
    avg_acc = np.mean(last_acc_table, axis=0)
    avg_avg_acc = np.mean(avg_acc)
    logging.info(f'average transfer acc: {transfer_avg_acc}')
    logging.info(f'average average acc: {avg_avg_acc}')
    logging.info(f'average last acc: {np.mean(last_acc_table[-1, :])}')


def evaluation(cfg, dataset_sequence, task_id, val_preprocess, last_acc_table, model, tokenizer):
    model.eval()
    
    all_texts = []
    for test_id, test_dataset in enumerate(dataset_sequence):
        if test_dataset == "cifar100":
            dataset = CIFAR100(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size,
                                location=cfg.data_dir)
        elif test_dataset == "mnist":
            dataset = MNIST(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size,
                            location=cfg.data_dir)
        else:
            dataset = build_dataset(test_dataset, cfg.data_dir, cfg.num_shots, val_preprocess)
        all_texts += [dataset.template[0].format(l) for l in dataset.classnames]
    with torch.no_grad():
        all_texts = tokenizer(all_texts).cuda()
        try:
            all_embeddings = model.encode_text(all_texts)
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings = all_embeddings.cuda()
        except:
            all_embeddings, _ = model.encode_text(all_texts)
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
            all_embeddings = all_embeddings.cuda()

    tested_cls_num = 0
    for test_id, test_dataset in enumerate(dataset_sequence):
        logging.info(f"Evaluating on dataset-{test_id + 1}: {test_dataset}")
        if test_dataset == "cifar100":
            test_set = CIFAR100(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size, 
                                location=cfg.data_dir)
        elif test_dataset == "mnist":
            test_set = MNIST(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size,
                                location=cfg.data_dir)
        else:
            test_set = build_dataset(test_dataset, cfg.data_dir, cfg.num_shots, val_preprocess)
            
        if test_dataset == "cifar100" or test_dataset == "mnist" or test_dataset == "imagenet" or test_dataset == "places365":
            test_loader = test_set.test_loader
        else:
            test_loader = build_data_loader(data_source=test_set.test, batch_size=cfg.batch_size, is_train=False,
                                            tfm=val_preprocess, shuffle=False)        

        top1, top5, test_num = 0.0, 0.0, 0.0

        for data in tqdm(test_loader, desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset}',
                        total=len(test_loader), unit='batch'):
            inputs, targets = data
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            test_num += inputs.size(0)
            targets += tested_cls_num

            with torch.no_grad():
                out, _, _ = model(inputs, None)
                out = out / out.norm(dim=-1, keepdim=True)
                outputs = model.logit_scale.exp() * out @ all_embeddings.t()

            # Zero-shot acc
            acc1, acc5 = cls_acc(outputs, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5

        top1, top5 = (top1 / test_num) * 100, (top5 / test_num) * 100
        logging.info(f"top-1 acc for dataset-{test_id + 1}: {test_dataset}: {top1}")

        last_acc_table[task_id, test_id] = top1

        tested_cls_num += len(test_set.classnames)
                
    logging.info(last_acc_table)
    if cfg.save is not None:
        results_path = cfg.save.split("/")[-1]
        save_path = os.path.join("./results", f"{results_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        outfile = os.path.join(save_path, f"xtail_results.npy")
        np.save(outfile, last_acc_table)
        logging.info(f"Results saved to {outfile}")


if __name__ == "__main__":
    args = parse_arguments()
    args.target_modules = LORA_MAPPING[args.target_modules_abbrev]

    logging.info(args)
    main(args)