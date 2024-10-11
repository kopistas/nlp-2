# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN-architecture with multiple convolutional layers and dropout
class CNNModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size, conv_channels=64, dropout=0.3, kernel_sizes=[3, 5]):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, hid_size)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hid_size, out_channels=conv_channels, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

        self.cat_encoder = nn.Linear(n_cat_features, hid_size)

        self.fc_out = nn.Linear(conv_channels * len(kernel_sizes) + hid_size, 1)

    def forward(self, batch):
        emb = self.embedding(batch['Title']).transpose(1, 2)  

        conv_results = [F.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]  
        text_features = torch.cat(conv_results, dim=1) 
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']))  

        final_out = torch.cat([text_features, cat_features], dim=1)  
        final_out = self.dropout(final_out)

        return self.fc_out(final_out).squeeze(-1)

class BiLSTMModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size=128, lstm_units=128, dropout=0.3, pooling='max'):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, hid_size)

        self.lstm = nn.LSTM(hid_size, lstm_units, batch_first=True, bidirectional=True)

        self.cat_encoder = nn.Linear(n_cat_features, hid_size)

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(2 * lstm_units + hid_size, 1)

        self.pooling = pooling

    def forward(self, batch):
        emb = self.embedding(batch['Title'])  
        
        lstm_out, _ = self.lstm(emb)  
        
        if self.pooling == 'max':
            lstm_out = lstm_out.max(dim=1)[0]  
        elif self.pooling == 'mean':
            lstm_out = lstm_out.mean(dim=1)  
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']))  

        final_out = torch.cat([lstm_out, cat_features], dim=1)  
        final_out = self.dropout(final_out)

        return self.fc_out(final_out).squeeze(-1)

# Mixed CNN + LSTM model
class CNNLSTMModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size, conv_channels=64, lstm_units=64, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, hid_size)

        self.title_conv = nn.Conv1d(in_channels=hid_size, out_channels=conv_channels, kernel_size=3)
        self.title_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.desc_lstm = nn.LSTM(hid_size, lstm_units, batch_first=True, bidirectional=True)
        
        self.cat_encoder = nn.Linear(n_cat_features, hid_size)
        
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(conv_channels + 2 * lstm_units + hid_size, 1)

    def forward(self, batch):
        title_emb = self.embedding(batch['Title']).transpose(1, 2) 
        title_out = F.relu(self.title_conv(title_emb))
        title_out = self.title_pool(title_out).max(dim=2)[0]  
        
        desc_emb = self.embedding(batch['FullDescription'])
        _, (desc_hidden, _) = self.desc_lstm(desc_emb)
        desc_out = torch.cat([desc_hidden[0], desc_hidden[1]], dim=1) 
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']))
        
        final_out = torch.cat([title_out, desc_out, cat_features], dim=1)
        final_out = self.dropout(final_out)
        
        return self.fc_out(final_out).squeeze(-1)

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size=128, nhead=4, num_layers=2, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, hid_size)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_size, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.cat_encoder = nn.Linear(n_cat_features, hid_size)
        
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hid_size * 2 + hid_size, 1)

    def forward(self, batch):
        title_emb = self.embedding(batch['Title']).transpose(0, 1)  
        desc_emb = self.embedding(batch['FullDescription']).transpose(0, 1)
    
        title_out = self.transformer(title_emb).mean(dim=0)  
        desc_out = self.transformer(desc_emb).mean(dim=0)
    
        cat_features = F.relu(self.cat_encoder(batch['Categorical']))
    
        final_out = torch.cat([title_out, desc_out, cat_features], dim=1)
        final_out = self.dropout(final_out)
        
        return self.fc_out(final_out).squeeze(-1)

# Pretrained Embeddings Model
class PretrainedEmbeddingsModel(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, hid_size):
        super(PretrainedEmbeddingsModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)  
        self.cat_encoder = nn.Linear(n_cat_features, hid_size)

        embedding_dim = embedding_matrix.shape[1]
        self.fc_out = nn.Linear(hid_size + 2 * embedding_dim, 1) 

    def forward(self, batch):
        title_emb = self.embedding(batch["Title"]) 
        desc_emb = self.embedding(batch["FullDescription"])  
        
        title_mean = title_emb.mean(dim=1)
        desc_mean = desc_emb.mean(dim=1)
        
        emb_out = torch.cat([title_mean, desc_mean], dim=1)  

        cat_features = F.relu(self.cat_encoder(batch['Categorical']))  

        final_out = torch.cat([emb_out, cat_features], dim=1)  

        return self.fc_out(final_out).squeeze(-1)

# CNN-RNN
class CNNRNNModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size, cnn_channels, kernel_sizes, rnn_hidden_size, rnn_num_layers, dropout=0.3):
        super(CNNRNNModel, self).__init__()

        self.embedding = nn.Embedding(n_tokens, hid_size)

        self.title_convs = nn.ModuleList([
            nn.Conv1d(in_channels=hid_size, out_channels=cnn_channels, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])

        self.desc_convs = nn.ModuleList([
            nn.Conv1d(in_channels=hid_size, out_channels=cnn_channels, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])

        self.title_rnn = nn.LSTM(input_size=cnn_channels * len(kernel_sizes), hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)

        self.desc_rnn = nn.LSTM(input_size=cnn_channels * len(kernel_sizes), hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)

        self.cat_encoder = nn.Linear(n_cat_features, rnn_hidden_size)

        self.fc_out = nn.Linear(rnn_hidden_size * 3, 1)  

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        title_emb = self.embedding(batch['Title']).transpose(1, 2)  
        title_conv_results = [F.relu(conv(title_emb)).transpose(1, 2) for conv in self.title_convs] 
        title_features = torch.cat(title_conv_results, dim=2)  

        _, (title_h_n, _) = self.title_rnn(title_features)  #
        title_rnn_features = title_h_n[-1]  

        desc_emb = self.embedding(batch['FullDescription']).transpose(1, 2)
        desc_conv_results = [F.relu(conv(desc_emb)).transpose(1, 2) for conv in self.desc_convs]
        desc_features = torch.cat(desc_conv_results, dim=2)

        _, (desc_h_n, _) = self.desc_rnn(desc_features)
        desc_rnn_features = desc_h_n[-1]

        cat_features = F.relu(self.cat_encoder(batch['Categorical']))

        combined_features = torch.cat([title_rnn_features, desc_rnn_features, cat_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.fc_out(combined_features).squeeze(-1)

        return output
    
    
class CNNRNNPoolingModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size, cnn_channels, kernel_sizes, rnn_hidden_size, rnn_num_layers, dropout=0.3):
        super(CNNRNNPoolingModel, self).__init__()

        self.embedding = nn.Embedding(n_tokens, hid_size)

        self.title_convs = nn.ModuleList([nn.Conv1d(hid_size, cnn_channels, ks, padding=ks // 2) for ks in kernel_sizes])
        self.desc_convs = nn.ModuleList([nn.Conv1d(hid_size, cnn_channels, ks, padding=ks // 2) for ks in kernel_sizes])

        self.title_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])
        self.desc_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])

        self.title_rnn = nn.LSTM(cnn_channels * len(kernel_sizes), rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.desc_rnn = nn.LSTM(cnn_channels * len(kernel_sizes), rnn_hidden_size, rnn_num_layers, batch_first=True)

        self.cat_encoder = nn.Linear(n_cat_features, rnn_hidden_size)

        self.fc_out = nn.Linear(rnn_hidden_size * 3, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        title_emb = self.embedding(batch['Title']).transpose(1, 2)
        title_conv_results = [F.relu(conv(title_emb)) for conv in self.title_convs]
        title_pooled_results = [pool(conv).transpose(1, 2) for conv, pool in zip(title_conv_results, self.title_pools)] 
        title_features = torch.cat(title_pooled_results, dim=2)

        _, (title_h_n, _) = self.title_rnn(title_features)
        title_rnn_features = title_h_n[-1]

        desc_emb = self.embedding(batch['FullDescription']).transpose(1, 2)
        desc_conv_results = [F.relu(conv(desc_emb)) for conv in self.desc_convs]
        desc_pooled_results = [pool(conv).transpose(1, 2) for conv, pool in zip(desc_conv_results, self.desc_pools)]  
        desc_features = torch.cat(desc_pooled_results, dim=2)

        _, (desc_h_n, _) = self.desc_rnn(desc_features)
        desc_rnn_features = desc_h_n[-1]

        cat_features = F.relu(self.cat_encoder(batch['Categorical']))

        combined_features = torch.cat([title_rnn_features, desc_rnn_features, cat_features], dim=1)
        combined_features = self.dropout(combined_features)

        # Final output
        output = self.fc_out(combined_features).squeeze(-1)

        return output
    
class CNNRNNPoolingDenseModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size, cnn_channels, kernel_sizes, rnn_hidden_size, rnn_num_layers, dense_after_cnn, dense_after_rnn, dropout=0.3):
        super(CNNRNNPoolingDenseModel, self).__init__()

        self.embedding = nn.Embedding(n_tokens, hid_size)

        self.title_convs = nn.ModuleList([nn.Conv1d(hid_size, cnn_channels, ks, padding=ks // 2) for ks in kernel_sizes])
        self.desc_convs = nn.ModuleList([nn.Conv1d(hid_size, cnn_channels, ks, padding=ks // 2) for ks in kernel_sizes])

        self.title_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])
        self.desc_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])

        self.title_dense = nn.Linear(cnn_channels * len(kernel_sizes), dense_after_cnn)
        self.desc_dense = nn.Linear(cnn_channels * len(kernel_sizes), dense_after_cnn)

        self.title_rnn = nn.LSTM(dense_after_cnn, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.desc_rnn = nn.LSTM(dense_after_cnn, rnn_hidden_size, rnn_num_layers, batch_first=True)

        self.cat_encoder = nn.Linear(n_cat_features, rnn_hidden_size)

        self.fc1 = nn.Linear(rnn_hidden_size * 3, dense_after_rnn)

        self.fc_out = nn.Linear(dense_after_rnn, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        # Process 'Title'
        title_emb = self.embedding(batch['Title']).transpose(1, 2)
        title_conv_results = [F.relu(conv(title_emb)) for conv in self.title_convs]
        title_pooled_results = [pool(conv).transpose(1, 2) for conv, pool in zip(title_conv_results, self.title_pools)]  # Apply pooling
        title_features = torch.cat(title_pooled_results, dim=2)
        title_features = F.relu(self.title_dense(title_features))  # Dense after CNN+Pooling

        _, (title_h_n, _) = self.title_rnn(title_features)
        title_rnn_features = title_h_n[-1]

        desc_emb = self.embedding(batch['FullDescription']).transpose(1, 2)
        desc_conv_results = [F.relu(conv(desc_emb)) for conv in self.desc_convs]
        desc_pooled_results = [pool(conv).transpose(1, 2) for conv, pool in zip(desc_conv_results, self.desc_pools)]  # Apply pooling
        desc_features = torch.cat(desc_pooled_results, dim=2)
        desc_features = F.relu(self.desc_dense(desc_features))  # Dense after CNN+Pooling

        _, (desc_h_n, _) = self.desc_rnn(desc_features)
        desc_rnn_features = desc_h_n[-1]

        cat_features = F.relu(self.cat_encoder(batch['Categorical']))

        combined_features = torch.cat([title_rnn_features, desc_rnn_features, cat_features], dim=1)

        combined_features = self.dropout(combined_features)

        combined_features = F.relu(self.fc1(combined_features))

        output = self.fc_out(combined_features).squeeze(-1)

        return output

class CNNRNNPoolingWithPretrainedEmbeddingsModel(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, cnn_channels, kernel_sizes, rnn_hidden_size, rnn_num_layers, freeze, dropout=0.3):
        super(CNNRNNPoolingWithPretrainedEmbeddingsModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze)

        embedding_dim = embedding_matrix.shape[1]  # Use embedding dimension directly from the matrix
        self.title_convs = nn.ModuleList([nn.Conv1d(embedding_dim, cnn_channels, ks, padding=ks // 2) for ks in kernel_sizes])
        self.desc_convs = nn.ModuleList([nn.Conv1d(embedding_dim, cnn_channels, ks, padding=ks // 2) for ks in kernel_sizes])

        self.title_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])
        self.desc_pools = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in kernel_sizes])

        self.title_rnn = nn.LSTM(cnn_channels * len(kernel_sizes), rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.desc_rnn = nn.LSTM(cnn_channels * len(kernel_sizes), rnn_hidden_size, rnn_num_layers, batch_first=True)

        self.cat_encoder = nn.Linear(n_cat_features, rnn_hidden_size)

        self.fc_out = nn.Linear(rnn_hidden_size * 3, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        title_emb = self.embedding(batch['Title']).transpose(1, 2)
        title_conv_results = [F.relu(conv(title_emb)) for conv in self.title_convs]
        title_pooled_results = [pool(conv).transpose(1, 2) for conv, pool in zip(title_conv_results, self.title_pools)] 
        title_features = torch.cat(title_pooled_results, dim=2)

        _, (title_h_n, _) = self.title_rnn(title_features)
        title_rnn_features = title_h_n[-1]

        desc_emb = self.embedding(batch['FullDescription']).transpose(1, 2)
        desc_conv_results = [F.relu(conv(desc_emb)) for conv in self.desc_convs]
        desc_pooled_results = [pool(conv).transpose(1, 2) for conv, pool in zip(desc_conv_results, self.desc_pools)]  
        desc_features = torch.cat(desc_pooled_results, dim=2)

        _, (desc_h_n, _) = self.desc_rnn(desc_features)
        desc_rnn_features = desc_h_n[-1]

        cat_features = F.relu(self.cat_encoder(batch['Categorical']))

        combined_features = torch.cat([title_rnn_features, desc_rnn_features, cat_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.fc_out(combined_features).squeeze(-1)

        return output
    
class MultiBiLSTMModel(nn.Module):
    def __init__(self, n_tokens, n_cat_features, embedding_dim=128, lstm_hidden_size=256, num_lstm_layers=2, dropout=0.3):
        super(MultiBiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=embedding_dim if i == 0 else lstm_hidden_size * 2,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for i in range(num_lstm_layers)
        ])
        
        self.scalar_parameters = nn.Parameter(torch.zeros(num_lstm_layers + 1))
        self.softmax = nn.Softmax(dim=0)
        
        self.cat_encoder = nn.Linear(n_cat_features, lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        input_size = lstm_hidden_size * 4 + lstm_hidden_size  
        self.fc_out = nn.Linear(input_size, 1)

    def forward(self, batch):
        # Embedding lookup
        title_emb = self.embedding(batch['Title'])  
        desc_emb = self.embedding(batch['FullDescription']) 
        
        title_hidden_states = [title_emb]
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(title_hidden_states[-1])
            title_hidden_states.append(lstm_out)
        
        desc_hidden_states = [desc_emb]
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(desc_hidden_states[-1])
            desc_hidden_states.append(lstm_out)
        
        scalar_weights = self.softmax(self.scalar_parameters)
        
        title_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, title_hidden_states))
        desc_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, desc_hidden_states))
        
        title_pooled = title_contextual_embeddings.mean(dim=1)  # Shape: (batch_size, lstm_hidden_size * 2)
        desc_pooled = desc_contextual_embeddings.mean(dim=1)  # Shape: (batch_size, lstm_hidden_size * 2)
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']))  
        
        combined_features = torch.cat([title_pooled, desc_pooled, cat_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        output = self.fc_out(combined_features).squeeze(-1)
        return output

class MultiBiLSTMEmbeddingModel(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, lstm_hidden_size=256, num_lstm_layers=2, freeze=False, dropout=0.3, padding_idx=None):
        super(MultiBiLSTMEmbeddingModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.lstm_hidden_size = lstm_hidden_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        
        self.embedding_projection = nn.Linear(embedding_dim, lstm_hidden_size * 2)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=lstm_hidden_size * 2,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for i in range(num_lstm_layers)
        ])
        
        self.scalar_parameters = nn.Parameter(torch.zeros(num_lstm_layers + 1))
        self.softmax = nn.Softmax(dim=0)
        
        self.cat_encoder = nn.Linear(n_cat_features, lstm_hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        input_size = lstm_hidden_size * 5  
        self.fc_out = nn.Linear(input_size, 1)

    def forward(self, batch):
        device = self.embedding.weight.device
        self.embedding_projection = self.embedding_projection.to(device)
        
        title_emb = self.embedding(batch['Title']).to(device)
        title_emb = self.embedding_projection(title_emb)
        
        desc_emb = self.embedding(batch['FullDescription']).to(device)
        desc_emb = self.embedding_projection(desc_emb)
        
        title_hidden_states = [title_emb]
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(title_hidden_states[-1])
            title_hidden_states.append(lstm_out)
        
        desc_hidden_states = [desc_emb]
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(desc_hidden_states[-1])
            desc_hidden_states.append(lstm_out)
        
        scalar_weights = self.softmax(self.scalar_parameters)
        
        title_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, title_hidden_states))
        desc_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, desc_hidden_states))
        
        title_pooled = title_contextual_embeddings.mean(dim=1) 
        desc_pooled = desc_contextual_embeddings.mean(dim=1)    
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))
        
        combined_features = torch.cat([title_pooled, desc_pooled, cat_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        output = self.fc_out(combined_features).squeeze(-1)
        return output

class CNNBiLSTMEmbeddingModel(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, lstm_hidden_size=256, num_lstm_layers=2, freeze=False, dropout=0.3, padding_idx=None):
        super(CNNBiLSTMEmbeddingModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.lstm_hidden_size = lstm_hidden_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=k, padding='same')
            for k in [3, 4, 5]
        ])
        total_conv_channels = 128 * len(self.conv_layers)
        
        self.conv_projection = nn.Linear(total_conv_channels, lstm_hidden_size * 2)
        
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm_input_size = lstm_hidden_size * 2  
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            )
        
        self.scalar_parameters = nn.Parameter(torch.zeros(num_lstm_layers + 1))
        self.softmax = nn.Softmax(dim=0)
        
        self.cat_encoder = nn.Linear(n_cat_features, lstm_hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        input_size = lstm_hidden_size * 4 + lstm_hidden_size 
        self.fc_out = nn.Linear(input_size, 1)
    
    def forward(self, batch):
        device = self.embedding.weight.device
        
        def process_text(text):
            emb = self.embedding(text).to(device) 
            emb = emb.permute(0, 2, 1) 
            
            conv_outputs = [F.relu(conv(emb)) for conv in self.conv_layers]
            conv_concat = torch.cat(conv_outputs, dim=1)  # (batch_size, total_conv_channels, seq_len)
            
            conv_concat = conv_concat.permute(0, 2, 1)
            
            projected_conv = self.conv_projection(conv_concat) 
            
            hidden_states = [projected_conv]
            input_to_lstm = projected_conv
            for lstm in self.lstm_layers:
                lstm_out, _ = lstm(input_to_lstm)
                hidden_states.append(lstm_out)
                input_to_lstm = lstm_out
            
            return hidden_states
        
        title_hidden_states = process_text(batch['Title'])
        desc_hidden_states = process_text(batch['FullDescription'])
        
        scalar_weights = self.softmax(self.scalar_parameters)
        
        title_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, title_hidden_states))
        desc_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, desc_hidden_states))
        
        title_pooled, _ = torch.max(title_contextual_embeddings, dim=1)
        desc_pooled, _ = torch.max(desc_contextual_embeddings, dim=1)
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))
        
        combined_features = torch.cat([title_pooled, desc_pooled, cat_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        output = self.fc_out(combined_features).squeeze(-1)
        return output

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  
        context = torch.sum(attn_weights * lstm_output, dim=1)  
        return context

class CNNBiLSTMEmbeddingAttentionModel(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, lstm_hidden_size=256, num_lstm_layers=2, freeze=False, dropout=0.3, padding_idx=None):
        super(CNNBiLSTMEmbeddingAttentionModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.lstm_hidden_size = lstm_hidden_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=k, padding='same')
            for k in [3, 4, 5]
        ])
        total_conv_channels = 128 * len(self.conv_layers)
        
        self.conv_projection = nn.Linear(total_conv_channels, lstm_hidden_size * 2)
        
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm_input_size = lstm_hidden_size * 2  
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            )
        
        self.scalar_parameters = nn.Parameter(torch.zeros(num_lstm_layers + 1))
        self.softmax = nn.Softmax(dim=0)
        
        self.attention = Attention(lstm_hidden_size)
        
        self.cat_encoder = nn.Linear(n_cat_features, lstm_hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        input_size = lstm_hidden_size * 4 + lstm_hidden_size  # 2 texts * lstm_hidden_size*2 + categorical features
        self.fc_out = nn.Linear(input_size, 1)
    
    def forward(self, batch):
        device = self.embedding.weight.device
        
        def process_text(text):
            emb = self.embedding(text).to(device)  
            emb = emb.permute(0, 2, 1)  
            
            conv_outputs = [F.relu(conv(emb)) for conv in self.conv_layers]
            conv_concat = torch.cat(conv_outputs, dim=1)  
            
            conv_concat = conv_concat.permute(0, 2, 1)
            
            projected_conv = self.conv_projection(conv_concat) 
            
            hidden_states = [projected_conv]
            input_to_lstm = projected_conv
            for lstm in self.lstm_layers:
                lstm_out, _ = lstm(input_to_lstm)
                hidden_states.append(lstm_out)
                input_to_lstm = lstm_out
            
            scalar_weights = self.softmax(self.scalar_parameters)
            combined_hidden_states = sum(w * h for w, h in zip(scalar_weights, hidden_states))
            
            context_vector = self.attention(combined_hidden_states)
            return context_vector
        
        title_context = process_text(batch['Title'])
        desc_context = process_text(batch['FullDescription'])
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))
        
        combined_features = torch.cat([title_context, desc_context, cat_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        output = self.fc_out(combined_features).squeeze(-1)
        return output

class CNNBiLSTMWithTFIDFModel(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, tfidf_size, lstm_hidden_size=256, num_lstm_layers=2, freeze=False, dropout=0.3, padding_idx=None):
        super(CNNBiLSTMWithTFIDFModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.lstm_hidden_size = lstm_hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_idx)

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=k, padding='same')
            for k in [3, 4, 5]
        ])
        total_conv_channels = 128 * len(self.conv_layers)

        self.conv_projection = nn.Linear(total_conv_channels, lstm_hidden_size * 2)

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_hidden_size * 2,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            )

        self.attention = nn.Linear(lstm_hidden_size * 2, 1)

        self.cat_encoder = nn.Linear(n_cat_features, lstm_hidden_size)

        self.tfidf_encoder = nn.Linear(tfidf_size, lstm_hidden_size)

        self.dropout = nn.Dropout(dropout)

        input_size = lstm_hidden_size * 4 + lstm_hidden_size * 2  # Adjusted for TF-IDF features
        self.fc_out = nn.Linear(input_size, 1)

    def forward(self, batch):
        device = self.embedding.weight.device

        def process_text(text):
            emb = self.embedding(text).to(device)
            emb = emb.permute(0, 2, 1)

            conv_outputs = [F.relu(conv(emb)) for conv in self.conv_layers]
            conv_concat = torch.cat(conv_outputs, dim=1)
            conv_concat = conv_concat.permute(0, 2, 1)

            projected_conv = self.conv_projection(conv_concat)

            lstm_output = projected_conv
            for lstm in self.lstm_layers:
                lstm_output, _ = lstm(lstm_output)

            attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
            context_vector = torch.sum(attn_weights * lstm_output, dim=1)

            return context_vector

        title_context = process_text(batch['Title'])
        desc_context = process_text(batch['FullDescription'])

        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))

        tfidf_features = F.relu(self.tfidf_encoder(batch['TFIDF']).to(device))

        combined_features = torch.cat([title_context, desc_context, cat_features, tfidf_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.fc_out(combined_features).squeeze(-1)
        return output
    
class FFNNWithTFIDFModel(nn.Module):
    def __init__(self, n_cat_features, tfidf_size, hidden_size=256, dropout=0.3):
        super(FFNNWithTFIDFModel, self).__init__()
        
        self.cat_encoder = nn.Linear(n_cat_features, hidden_size)
        
        self.tfidf_encoder = nn.Linear(tfidf_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_size * 2, 1)

    def forward(self, batch):
        device = next(self.parameters()).device

        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))

        tfidf_features = F.relu(self.tfidf_encoder(batch['TFIDF']).to(device))

        combined_features = torch.cat([cat_features, tfidf_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.fc_out(combined_features).squeeze(-1)
        return output
    
class TransformerWithTFIDFModel(nn.Module):
    def __init__(self, n_cat_features, tfidf_size, hidden_size=256, num_heads=4, num_layers=2, dropout=0.3):
        super(TransformerWithTFIDFModel, self).__init__()
        
        self.cat_encoder = nn.Linear(n_cat_features, hidden_size)
        
        self.tfidf_encoder = nn.Linear(tfidf_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_size * 2, 1) 

    def forward(self, batch):
        device = next(self.parameters()).device

        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))

        tfidf_features = F.relu(self.tfidf_encoder(batch['TFIDF']).to(device))

        tfidf_features = tfidf_features.unsqueeze(1)  
        tfidf_features = self.transformer_encoder(tfidf_features) 
        
        tfidf_features = tfidf_features.squeeze(1)

        combined_features = torch.cat([cat_features, tfidf_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.fc_out(combined_features).squeeze(-1)
        return output
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTMEmbeddingV2Model(nn.Module):
    def __init__(self, embedding_matrix, n_cat_features, lstm_hidden_size=128, num_lstm_layers=1, freeze=False, dropout=0.4, padding_idx=None):
        super(CNNBiLSTMEmbeddingV2Model, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.lstm_hidden_size = lstm_hidden_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=padding_idx)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=k, padding='same')
            for k in [3, 4, 5]
        ])
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(len(self.conv_layers))])
        total_conv_channels = 128 * len(self.conv_layers)
        
        self.conv_projection = nn.Linear(total_conv_channels, lstm_hidden_size * 2)
        
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm_input_size = lstm_hidden_size * 2 
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            )
        
        self.scalar_parameters = nn.Parameter(torch.zeros(num_lstm_layers + 1))
        self.softmax = nn.Softmax(dim=0)
        
        self.cat_encoder = nn.Linear(n_cat_features, lstm_hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        input_size = lstm_hidden_size * 4 + lstm_hidden_size
        self.fc_out = nn.Linear(input_size, 1)
    
    def forward(self, batch):
        device = self.embedding.weight.device
        
        def process_text(text):
            emb = self.embedding(text).to(device)  
            emb = emb.permute(0, 2, 1)  
            
            conv_outputs = [self.batch_norm_layers[i](F.relu(conv(emb))) for i, conv in enumerate(self.conv_layers)]
            conv_concat = torch.cat(conv_outputs, dim=1)  
            
            conv_concat = conv_concat.permute(0, 2, 1)
            
            projected_conv = self.conv_projection(conv_concat)  
            
            hidden_states = [projected_conv]
            input_to_lstm = projected_conv
            for lstm in self.lstm_layers:
                lstm_out, _ = lstm(input_to_lstm)
                hidden_states.append(lstm_out)
                input_to_lstm = lstm_out
            
            return hidden_states
        
        title_hidden_states = process_text(batch['Title'])
        desc_hidden_states = process_text(batch['FullDescription'])
        
        scalar_weights = self.softmax(self.scalar_parameters)
        
        title_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, title_hidden_states))
        desc_contextual_embeddings = sum(w * h for w, h in zip(scalar_weights, desc_hidden_states))
        
        title_pooled, _ = torch.max(title_contextual_embeddings, dim=1)
        desc_pooled, _ = torch.max(desc_contextual_embeddings, dim=1)
        
        cat_features = F.relu(self.cat_encoder(batch['Categorical']).to(device))
        
        combined_features = torch.cat([title_pooled, desc_pooled, cat_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        output = self.fc_out(combined_features).squeeze(-1)
        return output
