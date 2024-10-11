# pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import logging
import pandas as pd
import os
import numpy as np

class ModelPipeline:
    def __init__(self, models, data_train, data_val, token_to_id, categorical_vectorizer, categorical_columns,
                 UNK_IX, PAD_IX,
                 log_dir='logs', checkpoint_dir='checkpoints', early_stopping_patience=5,
                 device=torch.device('cpu'), batch_size=16, epochs=5, lr=1e-4):
        self.models = models
        self.data_train = data_train
        self.data_val = data_val
        self.token_to_id = token_to_id
        self.categorical_vectorizer = categorical_vectorizer
        self.categorical_columns = categorical_columns
        self.UNK_IX = UNK_IX
        self.PAD_IX = PAD_IX
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.results = []

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                                logging.StreamHandler()
                            ])
        self.logger = logging.getLogger(__name__)

    def generate_param_grid(self, param_grid):
        """Generates all combinations of hyperparameters."""
        import itertools
        fixed_params = {}
        grid_params = {}
        
        for key, value in param_grid.items():
            if isinstance(value, list) and key != 'embedding_matrix':  
                grid_params[key] = value
            else:  
                fixed_params[key] = value
        
        keys, values = zip(*grid_params.items()) if grid_params else ([], [])
        for v in itertools.product(*values):
            grid_combo = dict(zip(keys, v))
            yield {**fixed_params, **grid_combo}  
                
    def iterate_minibatches(self, data, batch_size):
        """Iterates over mini-batches."""
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = self.make_batch(data.iloc[indices[start:start + batch_size]])
            yield batch

    def evaluate_model(self, model, data, batch_size):
        """Evaluates the model: computes MAE and MSE."""
        squared_error = abs_error = num_samples = 0.0
        model.eval()
        with torch.no_grad():
            for batch in self.iterate_minibatches(data, batch_size):
                pred = model(batch)
                target = batch['Log1pSalary']
                squared_error += torch.sum(torch.square(pred - target)).item()
                abs_error += torch.sum(torch.abs(pred - target)).item()
                num_samples += len(pred)
        mse = squared_error / num_samples
        mae = abs_error / num_samples
        return mse, mae

    def make_batch(self, data):
        """Forms a batch of data."""
        from data_utils import as_matrix

        batch = {}
        batch["Title"] = as_matrix(data["Title"].values, PAD_IX=self.PAD_IX).astype(np.int64)
        batch["FullDescription"] = as_matrix(data["FullDescription"].values, PAD_IX=self.PAD_IX).astype(np.int64)
        batch['Categorical'] = self.categorical_vectorizer.transform(data[self.categorical_columns].to_dict(orient='records'))
        batch['Log1pSalary'] = data['Log1pSalary'].values.astype(np.float32)
        batch = {key: torch.tensor(value, dtype=torch.long if key in ["Title", "FullDescription"] else torch.float32, device=self.device) 
                for key, value in batch.items()}
        return batch

    def build_checkpoint_filename(self, model_name, hyperparams):
        """Creates a filename for saving the model checkpoint."""
        import hashlib
        hyperparams_str = '_'.join(f"{k}={v}" for k, v in sorted(hyperparams.items()) if k != 'embedding_matrix')
        filename = f"{model_name}_{hyperparams_str}.pt"
        # Hash the filename to prevent overly long filenames
        filename_hash = hashlib.md5(filename.encode()).hexdigest()
        return os.path.join(self.checkpoint_dir, f"{filename_hash}.pt")

    def train_and_evaluate(self):
        for model_name, model_class, param_grid in self.models:
            for hyperparams in self.generate_param_grid(param_grid):
                batch_size = hyperparams.get('batch_size', self.batch_size)
                model_checkpoint_path = self.build_checkpoint_filename(model_name, hyperparams)
                start_epoch = 0
                best_val_mae = float('inf')
                epochs_without_improvement = 0

                self.logger.info(f"Starting training for model: {model_name} with hyperparams: {hyperparams}")

                # Instantiate the model
                model_params = {k: v for k, v in hyperparams.items() if k not in ['batch_size', 'epochs']}
                model = model_class(**model_params).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                criterion = torch.nn.MSELoss(reduction='sum')

                # Check if there is a checkpoint
                if os.path.exists(model_checkpoint_path):
                    self.logger.info(f"Loading checkpoint for model {model_name} with hyperparams {hyperparams}")
                    checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_mae = checkpoint.get('best_val_mae', float('inf'))
                    epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
                    self.logger.info(f"Resuming training from epoch {start_epoch}")
                else:
                    self.logger.info(f"No checkpoint found for model {model_name} with hyperparams {hyperparams}. Starting fresh.")

                num_epochs = hyperparams.get('epochs', self.epochs)
                pbar_epochs = tqdm(range(start_epoch, num_epochs), desc=f"Training {model_name}", leave=True, mininterval=1.0)

                for epoch in pbar_epochs:
                    model.train()
                    total_loss = 0
                    
                    for i, batch in enumerate(self.iterate_minibatches(self.data_train, batch_size)):
                        pred = model(batch)
                        loss = criterion(pred, batch['Log1pSalary'])
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                    
                    mse, mae = self.evaluate_model(model, self.data_val, batch_size)

                    pbar_epochs.set_postfix({'Validation MAE': f'{mae:.5f}', 'Total Loss': f'{total_loss:.5f}'})
                    self.logger.info(f"Epoch {epoch + 1} - Validation MAE: {mae:.5f}, Total Loss: {total_loss:.5f}")
                    
                    # Save checkpoint at each epoch
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_mae': best_val_mae,
                        'epochs_without_improvement': epochs_without_improvement,
                        'hyperparams': hyperparams,
                        'model_name': model_name
                    }
                    torch.save(checkpoint, model_checkpoint_path)

                    if mae < best_val_mae:
                        best_val_mae = mae
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}. No improvement for {self.early_stopping_patience} epochs.")
                        pbar_epochs.close()
                        break

                result = {
                    'model_name': model_name,
                    'hyperparams': hyperparams,
                    'val_mae': best_val_mae
                }
                self.results.append(result)
                self.logger.info(f"Finished training for model: {model_name} with hyperparams: {hyperparams}. Best Validation MAE: {best_val_mae:.5f}")

        self.print_final_results()

    def print_final_results(self):
        """Prints final results for all models in tabular form."""
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by='val_mae')
        self.logger.info("\nFinal results:\n")
        self.logger.info(df_results)
        print(df_results)
