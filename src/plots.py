# src/plots.py
"""
Visualization functions for exploratory data analysis (EDA).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Global settings for plots
sns.set_theme(style="whitegrid")

def plot_target_distribution(df: pd.DataFrame, target_col: str):
    """
    Generates a bar plot (countplot) for the target variable's distribution.
    """
    plt.figure(figsize=(8, 5))
    
    # Calculate percentages
    total = len(df)
    ax = sns.countplot(x=target_col, data=df, palette="viridis")
    
    plt.title(f'Distribution of Target Variable ({target_col})', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add percentage labels
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 3,
                f'{100 * height / total:.1f}%',
                ha="center", va="bottom")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = None, corr_threshold: float = 0.8):
    """
    Plots a heatmap of the correlation matrix.
    
    - If target_col is provided, it only shows the correlation of features with the target.
    - If not, it shows the full matrix (can be very large).
    - Only annotates absolute correlations > corr_threshold for readability.
    """
    if target_col and target_col in df.columns:
        # Heatmap against the target variable only
        corr = df.corr()[[target_col]].sort_values(by=target_col, ascending=False)
        plt.figure(figsize=(10, 20))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm_r", vmin=-1, vmax=1)
        plt.title(f'Correlation with {target_col}', fontsize=16)
    else:
        # Full correlation matrix
        corr = df.corr()
        
        # Mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        plt.figure(figsize=(24, 18)) # Needs to be large
        sns.heatmap(corr, 
                    mask=mask, 
                    annot=corr.abs() > corr_threshold, # Only annotate high values
                    fmt=".1f", 
                    cmap='coolwarm_r', 
                    vmax=1, 
                    vmin=-1, 
                    center=0,
                    linewidths=.5, 
                    cbar_kws={"shrink": .5})
        plt.title(f'Feature Correlation Heatmap (Values > |{corr_threshold}|)', fontsize=20)
        
    plt.show()


def plot_feature_histograms(df: pd.DataFrame, features: list, n_cols: int = 5):
    """
    Plots histograms for a list of features in a grid.
    """
    n_rows = (len(features) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten() # Flatten to 1D for easy iteration

    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=20)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_feature_boxplots(df: pd.DataFrame, features: list):
    """
    Plots boxplots for a list of features in a single chart.
    Horizontally oriented for better readability.
    """
    if not features:
        print("No features provided for boxplots.")
        return
        
    plt.figure(figsize=(10, len(features) * 0.4)) # Adjust height dynamically
    sns.boxplot(data=df[features], orient='h', palette='Set2')
    plt.title('Feature Boxplots', fontsize=16)
    plt.xlabel('Value', fontsize=12)
    plt.show()