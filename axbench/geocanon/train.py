from argparse import ArgumentParser

import torch
from generative.nf import train_flow
from utils.activation_dataset import ActivationDataset
from utils.misc import get_optimizer_cls, get_model_args, make_activation_dataset_dir_path, make_path

def train(
        base_model_name,
        base_layer_name,
        base_dataset_name,
        base_dataset_train_split,
        base_dataset_test_split,
        base_dataset_config,
        activations_base_path,
        gen_model_name,
        gen_model_type,
        optimizer,
        lr,
        batch_size,
        epochs,
        save_freq,
        device
):
    train_dataset = ActivationDataset(
        model_name=base_model_name,
        layer=base_layer_name,
        dataset=base_dataset_name,
        split=base_dataset_train_split,
        dataset_config=base_dataset_config,
        base_dir=activations_base_path,
        device=device,
        dtype=torch.float32
    )
    test_dataset = ActivationDataset(
        model_name=base_model_name,
        layer=base_layer_name,
        dataset=base_dataset_name,
        split=base_dataset_test_split,
        dataset_config=base_dataset_config,
        base_dir=activations_base_path,
        device=device,
        dtype=torch.float32
    )
    optimizer, optimizer_args = get_optimizer_cls(optimizer)
    model_name, model_args = get_model_args(gen_model_name)
    save_dir=make_path(
                make_activation_dataset_dir_path(activations_base_path,
                                                base_dataset_name,
                                                base_dataset_config,
                                                base_dataset_train_split),
                base_model_name,
                base_layer_name,
                "gen_models",
                gen_model_name,
            )
    if gen_model_type == "nf":
        train_flow(
            train_data=train_dataset,
            test_data=test_dataset,
            optimizer_cls=optimizer,
            optimizer_args=optimizer_args,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            model=model_name,
            model_args=model_args,
            save_freq=save_freq,
            save_dir=save_dir,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model type: {gen_model_type}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the model whose activations are being learned")
    parser.add_argument("--base_layer_name", type=str, required=True, help="Name of the layer whose activations are being learned")
    parser.add_argument("--base_dataset_name", type=str, required=True, help="Name of the dataset whose activations are being learned")
    parser.add_argument("--base_dataset_train_split", type=str, default="train", help="The dataset split whose activations are being learned")
    parser.add_argument("--base_dataset_test_split", type=str, default="test", help="The dataset split to test learned activations")
    parser.add_argument("--base_dataset_config", type=str, default=None, help="The dataset configuration whose activations are being learned")
    parser.add_argument("--activations_base_path", type=str, default="/mnt/storage/yolcu/geocanon-activations", help="Path to the training data folder")
    parser.add_argument("--gen_model_name", type=str, required=True, help="Name of the generative model to train")
    parser.add_argument("--gen_model_type", choices=["nf"], required=True, help="Name of the generative model type")
    parser.add_argument("--optimizer", choices=["adam"], default="adam", help="Optimizer to use for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_freq", type=int, default=0, help="Frequency (in epochs) to save the model during training. 0 means only save at the end.")
    args = parser.parse_args()

    train(
        base_model_name=args.base_model_name,
        base_layer_name=args.base_layer_name,
        base_dataset_name=args.base_dataset_name,
        base_dataset_train_split=args.base_dataset_train_split,
        base_dataset_test_split=args.base_dataset_test_split,
        base_dataset_config=args.base_dataset_config,
        activations_base_path=args.activations_base_path,
        gen_model_name=args.gen_model_name,
        gen_model_type=args.gen_model_type,
        optimizer=args.optimizer,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_freq=args.save_freq,
        device="cuda:0"
    )
