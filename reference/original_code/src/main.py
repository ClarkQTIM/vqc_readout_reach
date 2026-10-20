import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset
from mednist_loader import get_retinamnist_dataloaders
from feature_extractor import get_feature_extractor
from quantum_models import draw_vqc_circuit, init_weights, build_vqc_circuit
from train_test import train_model, test_model

def main():
    parser = argparse.ArgumentParser(description="Run CNN-VQC Experiment (Single Config, No Noise)")

    # Dataset & image loading
    parser.add_argument('--dataset_name', type=str, default="seed")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_col', type=str, required=True)
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--split_col', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)

    # Architecture (single config)
    parser.add_argument('--n_qubits', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--single_qubit_op', type=str, required=True)
    parser.add_argument('--entanglement_op', type=str, required=True)
    parser.add_argument('--topology', type=str, required=True)

    # Training config
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.save_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load dataset
    print('Preparing to load dataset')
    if args.dataset_name == "retina":
        train_loader, val_loader, test_loader = get_retinamnist_dataloaders(
            img_size=(256, 256), batch_size=32, balance_classes=True,
            n_samples=250, num_workers=4
        )

    elif args.dataset_name == "iris":
        from iris_loader import get_iris_dataloaders
        train_loader, val_loader, test_loader = get_iris_dataloaders(batch_size=16)
        feature_extractor = None
        feature_dim = 4
        encoder = False

    else:
        train_dataset = ImageDataset(args.csv_path, args.image_dir, args.img_col, args.label_col, args.split_col, "train", True, 100)
        val_dataset = ImageDataset(args.csv_path, args.image_dir, args.img_col, args.label_col, args.split_col, "val", True, 100)
        test_dataset = ImageDataset(args.csv_path, args.image_dir, args.img_col, args.label_col, args.split_col, "test", True, 100)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Feature extractor and quantum head
    if args.dataset_name == "iris":
        feature_extractor = None
        feature_dim = 4
    else:
        feature_extractor, feature_dim = get_feature_extractor(extractor_type="resnet")

    weights = init_weights(
        n_qubits=args.n_qubits,
        num_layers=args.num_layers,
        single_qubit_op=args.single_qubit_op,
        entanglement_op=args.entanglement_op,
        topology=args.topology,
        encoder=encoder,
        input_dim=feature_dim if encoder else None
    )


    draw_vqc_circuit(
        n_qubits=args.n_qubits,
        num_layers=args.num_layers,
        weights=weights,
        save_path=os.path.join(args.save_dir, "vqc_circuit.png"),
        single_qubit_op=args.single_qubit_op,
        entanglement_op=args.entanglement_op,
        topology=args.topology
    )

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        feature_extractor=feature_extractor,
        encoder=weights["encoder_proj"],
        vqc_weights={
            "encoder_proj": weights["encoder_proj"],
            "sq_weights": weights["sq_weights"],
            "ent_weights": weights["ent_weights"]
        },
        vqc_callable=build_vqc_circuit(
            n_qubits=args.n_qubits,
            num_layers=args.num_layers,
            single_qubit_op=args.single_qubit_op,
            entanglement_op=args.entanglement_op,
            topology=args.topology,
            return_type="probs"
        ),
        save_dir=args.save_dir,
        n_epochs=args.n_epochs,
        encoder_lr=1e-4,
        vqc_lr=0.1
    )

    test_model(
        test_loader=test_loader,
        save_dir=args.save_dir,
        n_qubits=args.n_qubits,
        num_layers=args.num_layers,
        single_qubit_op=args.single_qubit_op,
        entanglement_op=args.entanglement_op,
        topology=args.topology,
        feature_extractor_type=None if args.dataset_name == "iris" else "resnet"
    )


if __name__ == "__main__":
    main()