"""
Updated main.py with optional STDB fine-tuning support.

Adds --fine_tune_snn and --snn_epochs flags for flexible hybrid approach:
- Fast mode: train_ann → calibrate → evaluate (30 min total)
- Thorough mode: train_ann → calibrate → train_snn (STDB) → evaluate (2-3 hrs total)
"""

import argparse
import torch
from pathlib import Path
from typing import Optional

from data import load_data
from ann_model import create_model
from train_ann import train_ann
from calibration import calibrate_and_compute_thresholds
from snn_model import create_snn, rate_code_input
from snn_training import finetune_snn  # NEW: STDB fine-tuning
from evaluate import evaluate_ann, evaluate_snn, print_evaluation_summary
from utils import (
    set_seed, setup_device, print_model_summary, load_checkpoint,
    print_training_config, save_checkpoint
)


def main_train_ann(args):
    """
    Train ANN from scratch.
    """
    print(f"\n{'='*70}")
    print("MODE: Train ANN")
    print(f"{'='*70}\n")
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Config
    config = {
        'Mode': 'Train ANN',
        'Dataset': args.dataset,
        'Epochs': args.epochs,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'Weight Decay': args.weight_decay,
        'Device': str(device),
        'Seed': args.seed,
    }
    print_training_config(config)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = create_model(architecture="simple", num_classes=10, device=device)
    
    # Train
    model, history = train_ann(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("Final Test Set Evaluation")
    print(f"{'='*70}\n")
    results = evaluate_ann(model, test_loader, device)
    
    print(f"\nTest Set Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Samples: {results['total_samples']}\n")
    
    # Save final model
    checkpoint_path = Path(args.checkpoint_dir) / "ann_trained.pth"
    save_checkpoint(model, None, args.epochs, results['overall_accuracy'], checkpoint_path)
    
    print(f"\n{'='*70}")
    print("ANN Training Complete!")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")


def main_calibrate_and_convert(args):
    """
    Calibrate thresholds and convert to SNN.
    """
    print(f"\n{'='*70}")
    print("MODE: Calibrate and Convert to SNN")
    print(f"{'='*70}\n")
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Load trained ANN
    model = create_model(architecture="simple", num_classes=10, device=device)
    checkpoint = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    
    print_model_summary(model, "Trained ANN")
    
    # Load calibration data
    print(f"\nLoading calibration dataset ({args.dataset})...")
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Limit calibration samples if specified
    if args.calibration_samples:
        from torch.utils.data import Subset
        calibration_set = Subset(val_loader.dataset, 
                                range(min(args.calibration_samples, len(val_loader.dataset))))
        calibration_loader = torch.utils.data.DataLoader(
            calibration_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        calibration_loader = val_loader
    
    # Calibrate thresholds
    thresholds = calibrate_and_compute_thresholds(
        model,
        calibration_loader,
        device,
        percentile=args.vth_percentile,
        scale=args.threshold_scale
    )
    
    # Create SNN
    print(f"\nCreating converted SNN...")
    snn_model = create_snn(model, thresholds, num_classes=10, device=device)
    print_model_summary(snn_model, "Converted SNN")
    
    # Save SNN
    snn_checkpoint_path = Path(args.checkpoint_dir) / "snn_converted.pth"
    torch.save({
        'snn_state_dict': snn_model.state_dict(),
        'thresholds': thresholds,
        'beta': 0.95,
    }, snn_checkpoint_path)
    print(f"SNN checkpoint saved: {snn_checkpoint_path}")
    
    print(f"\n{'='*70}")
    print("Calibration and Conversion Complete!")
    print(f"{'='*70}\n")


def main_evaluate(args):
    """
    Evaluate both ANN and SNN.
    """
    print(f"\n{'='*70}")
    print("MODE: Evaluate ANN and SNN")
    print(f"{'='*70}\n")
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Load test data
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load ANN
    print("Loading trained ANN...")
    ann_model = create_model(architecture="simple", num_classes=10, device=device)
    load_checkpoint(ann_model, args.checkpoint, device=device)
    ann_model = ann_model.to(device)
    
    # Evaluate ANN
    ann_results = evaluate_ann(ann_model, test_loader, device)
    
    # Load calibration data for threshold calibration
    print("\nLoading calibration dataset...")
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if args.calibration_samples:
        from torch.utils.data import Subset
        calibration_set = Subset(val_loader.dataset,
                                range(min(args.calibration_samples, len(val_loader.dataset))))
        calibration_loader = torch.utils.data.DataLoader(
            calibration_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        calibration_loader = val_loader
    
    # Calibrate and convert
    print("Calibrating thresholds...")
    thresholds = calibrate_and_compute_thresholds(
        ann_model,
        calibration_loader,
        device,
        percentile=args.vth_percentile,
        scale=args.threshold_scale
    )
    
    # Create SNN
    print("Creating SNN...")
    snn_model = create_snn(ann_model, thresholds, num_classes=10, device=device)
    
    # Evaluate SNN
    snn_results = evaluate_snn(snn_model, test_loader, device, time_steps=args.time_steps)
    
    # Print comparison
    print_evaluation_summary(ann_results, snn_results)


def main_finetune_snn(args):
    """
    Fine-tune SNN using STDB (NEW in Option C).
    """
    print(f"\n{'='*70}")
    print("MODE: Fine-tune SNN with STDB (Spike Timing Dependent Backpropagation)")
    print(f"{'='*70}\n")
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load trained ANN
    print("Loading trained ANN...")
    ann_model = create_model(architecture="simple", num_classes=10, device=device)
    load_checkpoint(ann_model, args.checkpoint, device=device)
    ann_model = ann_model.to(device)
    
    # Calibrate
    print("\nCalibrating thresholds...")
    if args.calibration_samples:
        from torch.utils.data import Subset
        calibration_set = Subset(val_loader.dataset,
                                range(min(args.calibration_samples, len(val_loader.dataset))))
        calibration_loader = torch.utils.data.DataLoader(
            calibration_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        calibration_loader = val_loader
    
    thresholds = calibrate_and_compute_thresholds(
        ann_model,
        calibration_loader,
        device,
        percentile=args.vth_percentile,
        scale=args.threshold_scale
    )
    
    # Create SNN
    print("\nCreating SNN from ANN...")
    snn_model = create_snn(ann_model, thresholds, num_classes=10, device=device)
    print_model_summary(snn_model, "Converted SNN (before STDB)")
    
    # Fine-tune with STDB
    print(f"\nFine-tuning SNN with STDB for {args.snn_epochs} epochs...")
    snn_model, stdb_history = finetune_snn(
        snn_model,
        train_loader,
        val_loader,
        epochs=args.snn_epochs,
        learning_rate=args.snn_learning_rate,
        lambda_reg=args.snn_lambda_reg,
        device=device,
        time_steps=args.time_steps,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save fine-tuned SNN
    snn_checkpoint_path = Path(args.checkpoint_dir) / "snn_stdb_finetuned.pth"
    torch.save(snn_model.state_dict(), snn_checkpoint_path)
    print(f"\nFine-tuned SNN saved: {snn_checkpoint_path}")


def main_full_pipeline(args):
    """
    Full pipeline: train → calibrate → [optionally fine-tune] → evaluate.
    """
    print(f"\n{'='*70}")
    if args.fine_tune_snn:
        print("MODE: Full Pipeline with STDB Fine-tuning")
    else:
        print("MODE: Full Pipeline (Direct Conversion Only)")
    print(f"{'='*70}\n")
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Config
    config = {
        'Mode': 'Full Pipeline',
        'Dataset': args.dataset,
        'Epochs': args.epochs,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'Time Steps': args.time_steps,
        'Fine-tune SNN': args.fine_tune_snn,
        'SNN Epochs': args.snn_epochs if args.fine_tune_snn else 'N/A',
        'Device': str(device),
        'Seed': args.seed,
    }
    print_training_config(config)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # ========== STEP 1: TRAIN ANN ==========
    print(f"\n{'#'*70}")
    print("STEP 1: Training ANN")
    print(f"{'#'*70}\n")
    
    ann_model = create_model(architecture="simple", num_classes=10, device=device)
    print_model_summary(ann_model, "ANN")
    
    ann_model, history = train_ann(
        ann_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # ========== STEP 2: CALIBRATE AND CONVERT ==========
    print(f"\n{'#'*70}")
    print("STEP 2: Calibrate and Convert to SNN")
    print(f"{'#'*70}\n")
    
    # Limit calibration samples
    if args.calibration_samples:
        from torch.utils.data import Subset
        calibration_set = Subset(val_loader.dataset,
                                range(min(args.calibration_samples, len(val_loader.dataset))))
        calibration_loader = torch.utils.data.DataLoader(
            calibration_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        calibration_loader = val_loader
    
    # Calibrate
    thresholds = calibrate_and_compute_thresholds(
        ann_model,
        calibration_loader,
        device,
        percentile=args.vth_percentile,
        scale=args.threshold_scale
    )
    
    # Convert
    snn_model = create_snn(ann_model, thresholds, num_classes=10, device=device)
    print_model_summary(snn_model, "Converted SNN")
    
    # ========== STEP 3 (OPTIONAL): STDB FINE-TUNING ==========
    if args.fine_tune_snn:
        print(f"\n{'#'*70}")
        print("STEP 3: Fine-tune SNN with STDB")
        print(f"{'#'*70}\n")
        
        snn_model, stdb_history = finetune_snn(
            snn_model,
            train_loader,
            val_loader,
            epochs=args.snn_epochs,
            learning_rate=args.snn_learning_rate,
            lambda_reg=args.snn_lambda_reg,
            device=device,
            time_steps=args.time_steps,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print(f"\n{'#'*70}")
        print("STEP 4: Evaluate and Compare")
        print(f"{'#'*70}\n")
    else:
        print(f"\n{'#'*70}")
        print("STEP 3 (SKIPPED): Fine-tuning disabled")
        print(f"{'#'*70}")
        print(f"\n{'#'*70}")
        print("STEP 3: Evaluate and Compare")
        print(f"{'#'*70}\n")
    
    # ========== FINAL: EVALUATE ==========
    ann_results = evaluate_ann(ann_model, test_loader, device)
    snn_results = evaluate_snn(snn_model, test_loader, device, time_steps=args.time_steps)
    
    print_evaluation_summary(ann_results, snn_results)
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving models and results...")
    print(f"{'='*70}\n")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save ANN
    ann_path = checkpoint_dir / "ann_final.pth"
    save_checkpoint(ann_model, None, args.epochs, ann_results['overall_accuracy'], ann_path)
    
    # Save SNN
    snn_path = checkpoint_dir / ("snn_final_stdb.pth" if args.fine_tune_snn else "snn_final.pth")
    torch.save({
        'snn_state_dict': snn_model.state_dict(),
        'thresholds': thresholds,
        'stdb_finetuned': args.fine_tune_snn,
    }, snn_path)
    
    print(f"ANN saved: {ann_path}")
    print(f"SNN saved: {snn_path}")
    
    # Save results summary
    import json
    results_summary = {
        'ann_accuracy': ann_results['overall_accuracy'],
        'snn_accuracy': snn_results['overall_accuracy'],
        'accuracy_drop': (ann_results['overall_accuracy'] - snn_results['overall_accuracy']) * 100,
        'time_steps': args.time_steps,
        'firing_rates': snn_results['average_firing_rates'],
        'stdb_finetuned': args.fine_tune_snn,
        'snn_epochs': args.snn_epochs if args.fine_tune_snn else None,
    }
    
    results_path = checkpoint_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results saved: {results_path}")
    
    print(f"\n{'='*70}")
    print("Full Pipeline Complete!")
    print(f"{'='*70}\n")


def create_parser():
    """
    Create argument parser with new STDB fine-tuning options.
    """
    parser = argparse.ArgumentParser(
        description="ANN-to-SNN Conversion Framework with Optional STDB Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode: Direct conversion only (30 min)
  python main.py --mode full_pipeline --epochs 50 --device cuda
  
  # Thorough mode: With STDB fine-tuning (2-3 hours)
  python main.py --mode full_pipeline --epochs 50 --fine_tune_snn --snn_epochs 5 --device cuda
  
  # Just fine-tune an existing SNN
  python main.py --mode finetune_snn --checkpoint checkpoints/ann_best.pth --snn_epochs 10
  
  # Evaluate after fine-tuning
  python main.py --mode evaluate --checkpoint checkpoints/ann_best.pth --time_steps 100
        """
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train_ann', 'calibrate_and_convert', 'finetune_snn', 'evaluate', 'full_pipeline'],
        default='full_pipeline',
        help='Execution mode'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'road_signs'],
        help='Dataset to use'
    )
    
    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (ANN)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (ANN)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='L2 regularization (ANN)'
    )
    
    # SNN Inference
    parser.add_argument(
        '--time_steps',
        type=int,
        default=100,
        help='Number of time steps for SNN inference'
    )
    parser.add_argument(
        '--vth_percentile',
        type=float,
        default=99.0,
        help='Percentile for threshold calibration'
    )
    parser.add_argument(
        '--threshold_scale',
        type=float,
        default=1.0,
        help='Threshold scaling factor'
    )
    
    # Calibration
    parser.add_argument(
        '--calibration_samples',
        type=int,
        default=1000,
        help='Number of samples for calibration'
    )
    
    # NEW: STDB Fine-tuning
    parser.add_argument(
        '--fine_tune_snn',
        action='store_true',
        help='Enable STDB fine-tuning of SNN weights (optional, slower but more accurate)'
    )
    parser.add_argument(
        '--snn_epochs',
        type=int,
        default=5,
        help='Number of STDB fine-tuning epochs (5-10 typical)'
    )
    parser.add_argument(
        '--snn_learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate for SNN STDB fine-tuning (lower than ANN)'
    )
    parser.add_argument(
        '--snn_lambda_reg',
        type=float,
        default=0.01,
        help='Sparsity regularization weight for STDB (0.001-0.1)'
    )
    
    # Checkpoints
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/ann_best.pth',
        help='Checkpoint path for loading'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory for saving checkpoints'
    )
    
    # Device and reproducibility
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Route to appropriate main function
    if args.mode == 'train_ann':
        main_train_ann(args)
    elif args.mode == 'calibrate_and_convert':
        main_calibrate_and_convert(args)
    elif args.mode == 'finetune_snn':
        main_finetune_snn(args)
    elif args.mode == 'evaluate':
        main_evaluate(args)
    elif args.mode == 'full_pipeline':
        main_full_pipeline(args)
    else:
        parser.print_help()
