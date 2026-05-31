import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import torch
from typing import List, Dict, Tuple, Any

# Import from src modules
from src.data_loader import load_event_log
from src.preprocessor import create_prefixes, build_activity_vocab, encode_and_pad
from src.baseline import MostFrequentClassBaseline, MeanRemainingTimeBaseline
from src.evaluate import evaluate_by_prefix_length
from src.model import ProcessTransformer
from src.train import train_model, create_data_loader
from src.myktybek_model import MyktbekModel
from src.myktybek_train import train_myktbek_model
from src.utils import split_by_time, save_results, plot_comparison


def run_single_dataset(data_path: str, dataset_name: str, output_dir: str, num_epochs: int = 10) -> None:
    """
    Run all experiments for a single dataset.
    
    Args:
        data_path (str): Path to the XES event log file.
        dataset_name (str): Name of the dataset for labeling results.
        output_dir (str): Directory to save results and plots.
        num_epochs (int): Number of training epochs for the transformer model.
    """
    print(f"\n{'='*60}")
    print(f"Running experiments for dataset: {dataset_name}")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1 — Load and preprocess
    print("Step 1: Loading and preprocessing data...")
    
    # Load event log
    df = load_event_log(data_path)
    print(f"  Loaded event log with {len(df)} events")
    
    # Generate prefixes
    prefixes_df = create_prefixes(df)
    print(f"  Generated {len(prefixes_df)} prefixes")
    
    # Split into train/test
    train_df, test_df = split_by_time(prefixes_df, test_ratio=0.2)
    print(f"  Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    
    # Build vocabulary from train set only
    vocab = build_activity_vocab(train_df)
    print(f"  Vocabulary size: {len(vocab)} activities")
    
    # Compute max_len from train set
    max_len = int(train_df['prefix_length'].max())
    print(f"  Maximum prefix length: {max_len}")
    
    # Encode train and test prefix_activities
    train_prefix_lists = train_df['prefix_activities'].tolist()
    test_prefix_lists = test_df['prefix_activities'].tolist()
    
    encoded_train = encode_and_pad(train_prefix_lists, vocab, max_len)
    encoded_test = encode_and_pad(test_prefix_lists, vocab, max_len)
    print(f"  Encoded train shape: {encoded_train.shape}")
    print(f"  Encoded test shape: {encoded_test.shape}")
    
    # Step 2 — Statistical baselines
    print("\nStep 2: Running statistical baselines...")
    
    # Activity prediction baseline
    mfc_baseline = MostFrequentClassBaseline()
    mfc_baseline.fit(train_df)
    mfc_activity_preds = mfc_baseline.predict(test_df)
    
    # Time prediction baseline
    mean_rt_baseline = MeanRemainingTimeBaseline()
    mean_rt_baseline.fit(train_df)
    mean_rt_preds = mean_rt_baseline.predict(test_df)
    
    # Evaluate statistical baselines
    baseline_eval_df = evaluate_by_prefix_length(
        test_df, 
        mfc_activity_preds, 
        mean_rt_preds
    )
    
    # Save baseline results
    save_results(baseline_eval_df, 'statistical_baseline', dataset_name, output_dir)
    print(f"  Saved statistical baseline results to {output_dir}/statistical_baseline_{dataset_name}.csv")
    
    # Step 3 — Train and evaluate attention-based model
    print("\nStep 3: Training Process Transformer model...")
    
    # Create DataLoader
    train_loader, mean_val, std_val = create_data_loader(
        train_df, 
        encoded_train, 
        vocab, 
        batch_size=32, 
        shuffle=True
    )
    print(f"  Created DataLoader with batches of size 32")
    
    # Instantiate model
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    model = ProcessTransformer(
        vocab_size=len(vocab), 
        embed_dim=64, 
        num_heads=4, 
        num_layers=2, 
        num_activities=len(vocab), 
        dropout=0.1, 
        max_len=max_len
    )
    
    # Train model
    epoch_losses = train_model(
        model, 
        train_loader, 
        num_epochs=num_epochs, 
        learning_rate=1e-3, 
        device=device
    )
    print(f"  Training completed. Final loss: {epoch_losses[-1]:.4f}")
    
    # Predict on test set
    print("\nStep 3b: Evaluating Process Transformer on test set...")
    model.eval()
    
    pt_activity_preds = []
    pt_time_preds = []
    
    # Reverse vocab for index → activity mapping
    idx_to_act = {v: k for k, v in vocab.items()}
    
    # Process test data in batches of 64
    test_batch_size = 64
    n_samples = len(encoded_test)
    
    with torch.no_grad():
        for i in range(0, n_samples, test_batch_size):
            batch_end = min(i + test_batch_size, n_samples)
            batch_x = encoded_test[i:batch_end].to(device)
            
            # Forward pass
            activity_logits, time_pred = model(batch_x)
            
            # Activity predictions: argmax of logits
            activity_indices = torch.argmax(activity_logits, dim=1).cpu().tolist()
            
            # Map indices back to activity names
            for idx in activity_indices:
                act_name = idx_to_act.get(idx, '')
                pt_activity_preds.append(act_name)
            
            # Time predictions: denormalize
            time_vals = time_pred.squeeze().cpu().tolist()
            if not isinstance(time_vals, list):
                time_vals = [time_vals]
            
            for val in time_vals:
                pred_seconds = val * std_val + mean_val
                pt_time_preds.append(pred_seconds)
    
    # Evaluate transformer model
    transformer_eval_df = evaluate_by_prefix_length(
        test_df, 
        pt_activity_preds, 
        pt_time_preds
    )
    
    # Save transformer results
    save_results(transformer_eval_df, 'process_transformer', dataset_name, output_dir)
    print(f"  Saved process transformer results to {output_dir}/process_transformer_{dataset_name}.csv")
    
    # Step 4 — Train and evaluate MyktbekModel
    print(f"\n--- Training MyktbekModel on {dataset_name} ---")

    myktbek_model = MyktbekModel(
        vocab_size=len(vocab),
        embed_dim=64,
        num_filters=128,
        num_activities=len(vocab),
        dropout=0.1
    )

    train_loader_mk, mean_val_mk, std_val_mk = create_data_loader(
        train_df, encoded_train, vocab, batch_size=32, shuffle=True
    )

    train_myktbek_model(
        myktbek_model,
        train_loader_mk,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        device=device
    )

    myktbek_model.eval()
    mk_activity_preds = []
    mk_time_preds = []

    with torch.no_grad():
        for i in range(0, len(encoded_test), 64):
            batch_tensor = encoded_test[i:i+64].to(device)
            act_logits, time_out = myktbek_model(batch_tensor)
            _, pred_indices = torch.max(act_logits, dim=1)
            mk_activity_preds.extend(pred_indices.tolist())
            denorm_time = (time_out.cpu().numpy() * std_val_mk) + mean_val_mk
            mk_time_preds.extend(denorm_time.flatten())

    mk_activity_preds_names = [idx_to_act.get(int(idx), '') for idx in mk_activity_preds]

    mk_eval_df = evaluate_by_prefix_length(test_df, mk_activity_preds_names, mk_time_preds)
    save_results(mk_eval_df, 'myktbek_cnn', dataset_name, output_dir)
    print(f"  Saved MyktbekModel results to {output_dir}/myktbek_cnn_{dataset_name}.csv")


    # Step 4 — Plot comparisons
    print("\nStep 4: Generating comparison plots...")
    
    # Build results dictionary for accuracy
    results_dict_accuracy = {
        'Statistical Baseline': baseline_eval_df,
        'ProcessTransformer': transformer_eval_df,
        'MyktbekModel': mk_eval_df,
    }

    
    # Build results dictionary for MAE (converted to days)
    results_dict_mae_days = {}
    for name, df in results_dict_accuracy.items():
        df_copy = df.copy()
        df_copy['mae'] = df_copy['mae'] / 86400  # Convert seconds to days
        results_dict_mae_days[name] = df_copy
    
    # Plot accuracy comparison
    plot_comparison(results_dict_accuracy, metric='accuracy', dataset_name=dataset_name, output_dir=output_dir)
    print(f"  Saved accuracy comparison plot to {output_dir}/{dataset_name}_accuracy_comparison.png")
    
    # Plot MAE comparison (in days)
    plot_comparison(results_dict_mae_days, metric='mae', dataset_name=dataset_name, output_dir=output_dir)
    print(f"  Saved MAE comparison plot (days) to {output_dir}/{dataset_name}_mae_comparison.png")
    
    print(f"\n✓ Completed experiments for {dataset_name}\n")


def main() -> None:
    """Main function to run experiments on all datasets."""
    
    # Define datasets and parameters
    datasets = [
        ('data/BPI_Challenge_2012.xes.gz', 'bpi2012'),
        ('data/BPI_Challenge_2017.xes.gz', 'bpi2017')
    ]
    
    output_dir = 'experiments/results'
    num_epochs = 10
    
    # Run experiments for each dataset
    for i, (data_path, dataset_name) in enumerate(datasets):
        if i > 0:
            print("\n" + "="*60)
            print("Separator between runs")
            print("="*60 + "\n")
        
        try:
            run_single_dataset(data_path, dataset_name, output_dir, num_epochs)
        except FileNotFoundError as e:
            print(f"\n✗ Error: Dataset file not found at {data_path}")
            print(f"  Skipping {dataset_name}.\n")
        except Exception as e:
            print(f"\n✗ Error running {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()