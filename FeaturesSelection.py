import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def calculate_vif(df):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the features for which VIF needs to be calculated.
        Ensure all features are numerical.

    Returns:
    --------
    vif_data : pandas.DataFrame
        A DataFrame containing two columns:
        - 'Feature': Names of the features.
        - 'VIF': Variance Inflation Factor values for each feature.
    """
    # Ensure all columns are numeric
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Create a DataFrame to store VIF results
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns
    vif_data["VIF"] = [round(variance_inflation_factor(df_numeric.values, i), 2)
                       for i in range(df_numeric.shape[1])]
    return vif_data


def remove_intercolinarity(X_train, threshold=15):
    """
    Iteratively remove features with high multicollinearity based on the Variance Inflation Factor (VIF).

    Parameters:
    -----------
    X_train : pandas.DataFrame
        The input training dataset containing features.
    threshold : int or float, optional
        The VIF threshold above which features will be removed. Default is 15.

    Returns:
    --------
    vif_results : pandas.DataFrame
        The final DataFrame of VIF values for features after intercollinearity has been reduced.
    """
    # Create a copy of the training data
    X_train_bis = X_train.copy()

    # Calculate initial VIF for all features
    vif_results = calculate_vif(X_train_bis)
    vif_results = vif_results.set_index("Feature")
    
    # Identify the feature with the highest VIF
    name = vif_results.sort_values("VIF", ascending=False).iloc[0].name
    value = vif_results.sort_values("VIF", ascending=False).iloc[0].values[0]
    
    # Iteratively remove features with VIF higher than the threshold
    while value > threshold:
        
        # Recalculate VIF
        vif_results = calculate_vif(X_train_bis)
        vif_results = vif_results.set_index("Feature")
        
        # Identify the feature with the highest VIF
        name = vif_results.sort_values("VIF", ascending=False).iloc[0].name
        value = vif_results.sort_values("VIF", ascending=False).iloc[0].values[0]
    
        # Remove the feature with the highest VIF
        del X_train_bis[name]

    # Return the final VIF DataFrame
    return vif_results


def correlation_graphs(df):
    """
    Generate two heatmaps side by side to visualize Pearson and Spearman correlations.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing numerical features to compute correlations.

    Returns:
    --------
    None
        Displays a figure with two heatmaps:
        - Left: Pearson correlation heatmap.
        - Right: Spearman correlation heatmap.
    """
    # Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))  # Adjust size for proper spacing
    
    # Heatmap for Pearson correlation
    sns.heatmap(df.corr(), fmt=".2f", cmap="coolwarm", annot=True, vmin=-1, vmax=1, ax=ax1)
    ax1.set_title("Pearson Correlation")
    
    # Heatmap for Spearman correlation
    sns.heatmap(df.corr(method="spearman"), fmt=".2f", cmap="coolwarm", annot=True, vmin=-1, vmax=1, ax=ax2)
    ax2.set_title("Spearman Correlation")
    
    # Display the two heatmaps side by side
    plt.tight_layout()
    plt.show()