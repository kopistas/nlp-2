import torch
import re
import pandas as pd
import matplotlib.pyplot as plt

def check_set_gpu(override=None):
    if override is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True

        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device

def parse_logs_and_plot_mae(log_file_path, mae_threshold=10000):
    model_name_pattern = r"Starting training for model: (\w+Model)"
    epoch_mae_pattern = r"Epoch (\d+) - Validation MAE: ([\d.]+)"
    final_mae_pattern = r"Finished training for model: (\w+Model) .* Best Validation MAE: ([\d.]+)"
    early_stopping_pattern = r"Early stopping at epoch (\d+)"
    
    data = []

    with open(log_file_path, "r") as file:
        current_model = None
        for line in file:
            try:
                model_match = re.search(model_name_pattern, line)
                if model_match:
                    current_model = model_match.group(1)

                epoch_match = re.search(epoch_mae_pattern, line)
                if epoch_match and current_model:
                    epoch = int(epoch_match.group(1))
                    val_mae = float(epoch_match.group(2))
                    data.append({"model": current_model, "epoch": epoch, "val_mae": val_mae})

                early_stop_match = re.search(early_stopping_pattern, line)
                if early_stop_match and current_model:
                    early_stop_epoch = int(early_stop_match.group(1))
                    data.append({"model": current_model, "epoch": early_stop_epoch, "val_mae": None})

                final_match = re.search(final_mae_pattern, line)
                if final_match:
                    model = final_match.group(1)
                    best_val_mae = float(final_match.group(2))
                    data.append({"model": model, "epoch": "Best", "val_mae": best_val_mae})

            except (ValueError, AttributeError) as e:
                continue

    df = pd.DataFrame(data)

    df = df[pd.to_numeric(df['val_mae'], errors='coerce').notnull()]

    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')

    model_epoch_counts = df.groupby('model')['epoch'].nunique()
    valid_models = model_epoch_counts[model_epoch_counts >= 10].index
    df = df[df['model'].isin(valid_models)]

    df = df[df['val_mae'] < mae_threshold]

    plt.figure(figsize=(10, 6))
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        plt.plot(model_data['epoch'], model_data['val_mae'], label=model_name)

    plt.yscale('log')

    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE (Log Scale)')
    plt.title('Validation MAE per Epoch for Models with 10 or More Epochs (Log Scale, Filtered)')
    plt.legend()
    plt.grid(True)

    plt.show()