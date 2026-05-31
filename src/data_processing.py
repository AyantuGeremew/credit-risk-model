import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_dataset_shape(df):
    """
    Returns the number of rows and columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    tuple: (rows, columns)
    """
    rows, cols = df.shape
    return rows, cols

def get_data_types(df):
    """
    Returns the data type of each column.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    -------
    pd.DataFrame
        DataFrame containing column names and their data types
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values
    })

def calculate_central_tendency(df):
    """
    Calculate mean, median, and mode for all numerical columns.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    -------
    pd.DataFrame
        Summary of central tendency measures
    """
    
    numeric_df = df.select_dtypes(include=['number'])

    summary = pd.DataFrame({
        'Mean': numeric_df.mean(),
        'Median': numeric_df.median(),
        'Mode': numeric_df.mode().iloc[0]
    })

    return summary

def calculate_dispersion(df):
    """
    Calculate dispersion measures for numerical columns.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    -------
    pd.DataFrame
        Dispersion statistics for numerical columns.
    """
    
    numeric_df = df.select_dtypes(include='number')

    dispersion = pd.DataFrame({
        'Range': numeric_df.max() - numeric_df.min(),
        'Variance': numeric_df.var(),
        'Std_Deviation': numeric_df.std(),
        'IQR': numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    })

    return dispersion

def distribution_shape(df):
    """
    Calculate skewness and kurtosis for all numerical columns.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    -------
    pd.DataFrame
        Distribution shape statistics.
    """
    
    numeric_df = df.select_dtypes(include='number')

    shape_summary = pd.DataFrame({
        'Skewness': numeric_df.skew(),
        'Kurtosis': numeric_df.kurt()
    })

    return shape_summary

def plot_histograms(df, bins=30):
    """
    Plot histograms with KDE for all numerical features.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame
    bins : int
        Number of histogram bins
    """

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))

        sns.histplot(
            data=df,
            x=col,
            bins=bins,
            kde=True
        )

        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def calculate_skewness(df):
    """
    Calculate skewness for numerical columns.
    """

    numeric_df = df.select_dtypes(include='number')

    skew_df = pd.DataFrame({
        'Feature': numeric_df.columns,
        'Skewness': numeric_df.skew().values
    })

    return skew_df.sort_values(
        by='Skewness',
        key=abs,
        ascending=False
    )

def plot_boxplots(df):
    """
    Plot boxplots for all numerical features.
    """

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 3))

        sns.boxplot(x=df[col])

        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()

def analyze_distributions(df, bins=30):
    """
    Analyze numerical feature distributions.
    Includes:
    - Histogram + KDE
    - Boxplot
    - Skewness
    """

    numeric_cols = df.select_dtypes(include='number').columns

    skewness_summary = []

    for col in numeric_cols:

        skewness_summary.append({
            "Feature": col,
            "Skewness": df[col].skew()
        })

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4)
        )

        # Histogram + KDE
        sns.histplot(
            df[col],
            bins=bins,
            kde=True,
            ax=axes[0]
        )

        axes[0].set_title(f'Distribution of {col}')

        # Boxplot
        sns.boxplot(
            x=df[col],
            ax=axes[1]
        )

        axes[1].set_title(f'Boxplot of {col}')

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(skewness_summary)

def get_categorical_columns(df):
    """
    Return a list of categorical columns.
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()
def category_frequency(df, column):
    """
    Calculate frequency and percentage distribution
    for a categorical feature.
    """

    frequency = df[column].value_counts(dropna=False)
    percentage = round(
        (frequency / len(df)) * 100,
        2
    )

    summary = pd.DataFrame({
        "Frequency": frequency,
        "Percentage (%)": percentage
    })

    return summary

def categorical_feature_analysis(df, top_n=10, plot=True):
    """
    Analyze categorical feature distributions in one function.

    Provides:
    - Frequency count
    - Percentage distribution
    - Number of unique categories (variability)
    - Top categories
    - Optional visualization

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataset
    top_n : int
        Number of top categories to display/plot
    plot : bool
        Whether to plot distributions

    Returns:
    -------
    pd.DataFrame
        Summary of categorical feature analysis
    """

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    results = []

    for col in categorical_cols:

        freq = df[col].value_counts(dropna=False)
        percent = (freq / len(df)) * 100

        results.append({
            "Feature": col,
            "Unique_Values": df[col].nunique(),
            "Top_Category": freq.index[0],
            "Top_Category_Count": freq.iloc[0],
            "Most_Common_Percentage": percent.iloc[0]
        })

        # Plotting
        if plot:
            plt.figure(figsize=(8, 4))

            sns.countplot(
                data=df,
                x=col,
                order=freq.index[:top_n]
            )

            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.xlabel(col)
            plt.ylabel("Count")

            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)

def correlation_matrix(df):
    """
    Compute correlation matrix for numerical features only.
    """
    numeric_df = df.select_dtypes(include='number')
    return numeric_df.corr()

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for numerical features.
    """
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.show()

def high_correlation_pairs(df, threshold=0.7):
    """
    Find strongly correlated feature pairs.
    """
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    pairs = []

    for i in range(len(corr.columns)):
        for j in range(i):
            value = corr.iloc[i, j]

            if abs(value) >= threshold:
                pairs.append({
                    "Feature_1": corr.columns[i],
                    "Feature_2": corr.columns[j],
                    "Correlation": value
                })

    return pd.DataFrame(pairs)

def correlation_analysis(df, threshold=0.7):
    """
    Complete correlation analysis:
    - Correlation matrix
    - Heatmap
    - Strong correlations
    """

    print("\n🔹 Numerical Features:")
    numeric_df = df.select_dtypes(include='number')
    print(numeric_df.columns.tolist())

    print("\n🔹 Correlation Matrix:")
    corr = correlation_matrix(df)
    print(corr)

    print("\n🔹 High Correlation Pairs:")
    high_corr = high_correlation_pairs(df, threshold)
    print(high_corr)

    print("\n🔹 Heatmap:")
    plot_correlation_heatmap(df)

    return corr, high_corr

def missing_value_analysis(df, threshold=0.3, plot=True):
    """
    Complete missing value analysis in one function:
    - Missing count & percentage
    - Columns with missing values
    - Imputation strategy suggestion
    - Optional visualization

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataset
    threshold : float
        Threshold for high missing values (default 30%)
    plot : bool
        Whether to show heatmap

    Returns:
    -------
    pd.DataFrame
        Missing value summary with recommended strategies
    """

    # Step 1: Missing summary
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percent
    }).sort_values(by="Missing_Count", ascending=False)

    # Step 2: Strategy decision
    strategies = []

    for col in summary.index:
        percent = summary.loc[col, "Missing_Percentage"]

        if percent == 0:
            strategy = "No missing values"
        elif percent < 5:
            strategy = "Drop rows or simple imputation"
        elif percent <= 30:
            strategy = "Mean/Median/Mode imputation"
        else:
            strategy = "Drop column or advanced imputation (KNN/MICE)"

        strategies.append(strategy)

    summary["Recommended_Strategy"] = strategies

    # Step 3: Print key insights
    print("\n🔹 Missing Value Summary:")
    print(summary)

    print("\n🔹 Columns with Missing Values:")
    print(summary[summary["Missing_Count"] > 0].index.tolist())

    # Step 4: Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()

    return summary

def detect_outliers_iqr(df):
    """
    Detect outliers using the IQR method for numerical features.

    Returns a DataFrame showing:
    - Q1, Q3
    - IQR
    - Lower and Upper bounds
    - Number of outliers
    """

    numeric_df = df.select_dtypes(include='number')

    results = []

    for col in numeric_df.columns:

        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]

        results.append({
            "Feature": col,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": len(outliers)
        })

    return pd.DataFrame(results)

def plot_all_boxplots(df):
    """
    Boxplots for all numerical features.
    """

    numeric_df = df.select_dtypes(include='number')

    for col in numeric_df.columns:

        plt.figure(figsize=(6, 4))

        sns.boxplot(x=numeric_df[col])

        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)

        plt.show()

def outlier_analysis(df):
    """
    Full outlier analysis:
    - Detect outliers (IQR method)
    - Visualize boxplots
    """

    print("\n🔹 Outlier Detection Summary (IQR Method):")
    summary = detect_outliers_iqr(df)
    print(summary)

    print("\n🔹 Boxplots for Numerical Features:")
    plot_all_boxplots(df)

    return summary