import torch
import torch.nn as nn
import torch.nn.functional as F


class MyktbekModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_activities=None,
                 filter_sizes=[3, 5, 7], num_filters=128, dropout=0.1):
        super(MyktbekModel, self).__init__()

        if num_activities is None:
            num_activities = vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Three parallel convolutional branches (kernel sizes 3, 5, 7)
        self.conv_branch_1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                     kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )

        self.conv_branch_2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                     kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )

        self.conv_branch_3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                     kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )

        # Global max pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Shared feature representation
        total_features = num_filters * len(filter_sizes)
        self.shared_fc = nn.Linear(total_features, 256)
        self.shared_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

        # Output Head 1: Activity Classification
        self.activity_head = nn.Linear(256, num_activities)

        # Output Head 2: Time Regression (predicts normalized time)
        self.time_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)

        # Parallel convolutional branches
        branch_1_out = self.conv_branch_1(embedded)
        branch_2_out = self.conv_branch_2(embedded)
        branch_3_out = self.conv_branch_3(embedded)

        # Global max pooling
        pooled_1 = self.global_pool(branch_1_out).squeeze(2)
        pooled_2 = self.global_pool(branch_2_out).squeeze(2)
        pooled_3 = self.global_pool(branch_3_out).squeeze(2)

        # Concatenate features
        combined = torch.cat([pooled_1, pooled_2, pooled_3], dim=1)

        # Shared FC layer
        shared = self.shared_fc(combined)
        shared = self.shared_bn(shared)
        shared = F.relu(shared)
        shared = self.dropout(shared)

        # Two output heads
        activity_logits = self.activity_head(shared)
        remaining_time_normalized = self.time_head(shared)

        return activity_logits, remaining_time_normalized