import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import *
from xverse.transformer import WOE


from sklearn.preprocessing import StandardScaler

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

#-----------------------------------------------
# Create Aggregate Features
#-------------------------------------------

def calculate_total_transaction_amount(df, customer_col, amount_col):
    """
    Calculate the total transaction amount for each customer.

    Parameters:
    ----------
    df : pd.DataFrame
        Input transaction data.
    customer_col : str
        Column containing customer IDs.
    amount_col : str
        Column containing transaction amounts.
    Returns:
    -------
    pd.DataFrame
        DataFrame with total transaction amount per customer.
    """
    
    total_amount = (
        df.groupby(customer_col)[amount_col]
        .sum()
        .reset_index()
        .rename(columns={amount_col: "Total_Transaction_Amount"})
    )
    
    return total_amount

def calculate_average_transaction_amount(df, customer_col, amount_col):
    """
    Calculate the average transaction amount for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset.
    customer_col : str
        Customer ID column name.
    amount_col : str
        Transaction amount column name.

    Returns
    -------
    pd.DataFrame
        DataFrame containing average transaction amount per customer.
    """

    avg_amount = (
        df.groupby(customer_col)[amount_col]
        .mean()
        .reset_index()
        .rename(columns={amount_col: "Average_Transaction_Amount"})
    )

    return avg_amount

def calculate_transaction_count(df, customer_col):
    """
    Calculate the number of transactions for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset.
    customer_col : str
        Customer ID column name.

    Returns
    -------
    pd.DataFrame
        DataFrame containing transaction count per customer.
    """

    transaction_count = (
        df.groupby(customer_col)
        .size()
        .reset_index(name="Transaction_Count")
    )

    return transaction_count

def calculate_transaction_amount_std(df, customer_col, amount_col):
    """
    Calculate standard deviation of transaction amounts per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset
    customer_col : str
        Customer ID column
    amount_col : str
        Transaction amount column

    Returns
    -------
    pd.DataFrame
        Standard deviation of transaction amounts per customer
    """

    std_df = (
        df.groupby(customer_col)[amount_col]
        .std()
        .reset_index()
        .rename(columns={amount_col: "Transaction_Amount_Std"})
    )

    return std_df

#-----------------------------------------------
# Extract Features
#-------------------------------------------

def extract_transaction_hour(df, datetime_col):
    """
    Extract hour of the day from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset
    datetime_col : str
        Column containing transaction datetime

    Returns
    -------
    pd.DataFrame
        DataFrame with CustomerId (if exists) and Transaction_Hour
    """

    df_copy = df.copy()

    # Ensure datetime format
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors="coerce")

    # Extract hour
    df_copy["Transaction_Hour"] = df_copy[datetime_col].dt.hour

    return df_copy

def add_transaction_hour_feature(df, datetime_col):
    """
    Add transaction hour feature to original dataset.
    """

    df = extract_transaction_hour(df, datetime_col)
    return df

def extract_transaction_day_of_month(df, datetime_col):
    """
    Extract day of the month from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset
    datetime_col : str
        Column containing transaction datetime

    Returns
    -------
    pd.DataFrame
        DataFrame with a new feature: Transaction_Day_of_Month
    """

    df_copy = df.copy()

    # Convert to datetime safely
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors="coerce")

    # Extract day of month (1–31)
    df_copy["Transaction_Day_of_Month"] = df_copy[datetime_col].dt.day

    return df_copy

def add_transaction_day_of_month_feature(df, datetime_col):
    """
    Add transaction day-of-month feature to the dataset.
    """

    return extract_transaction_day_of_month(df, datetime_col)

def extract_transaction_month(df, datetime_col):
    """
    Extract month from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset
    datetime_col : str
        Column containing transaction datetime

    Returns
    -------
    pd.DataFrame
        DataFrame with Transaction_Month feature added
    """

    df_copy = df.copy()

    # Convert to datetime safely
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors="coerce")

    # Extract month (1–12)
    df_copy["Transaction_Month"] = df_copy[datetime_col].dt.month

    return df_copy

def add_transaction_month_feature(df, datetime_col):
    """
    Add transaction month feature to dataset.
    """

    return extract_transaction_month(df, datetime_col)

def extract_transaction_year(df, datetime_col):
    """
    Extract year from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset
    datetime_col : str
        Column containing transaction datetime

    Returns
    -------
    pd.DataFrame
        DataFrame with Transaction_Year feature added
    """

    df_copy = df.copy()

    # Convert to datetime safely
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors="coerce")

    # Extract year (e.g., 2023, 2024)
    df_copy["Transaction_Year"] = df_copy[datetime_col].dt.year

    return df_copy

def add_transaction_year_feature(df, datetime_col):
    """
    Add transaction year feature to dataset.
    """

    return extract_transaction_year(df, datetime_col)

#-----------------------------------------------
# Encode Categorical Variables
#-------------------------------------------  

def one_hot_encode_column(df, column_name, drop_first=False):
    """
    Perform one-hot encoding on a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    column_name : str
        Name of categorical column to encode
    drop_first : bool, default=False
        Whether to drop first category (to avoid multicollinearity)

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded columns
    """

    df_copy = df.copy()

    encoded_df = pd.get_dummies(
        df_copy[column_name],
        prefix=column_name,
        drop_first=drop_first
    )

    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    df_copy.drop(columns=[column_name], inplace=True)

    return df_copy

def one_hot_encode_multiple(df, columns, drop_first=False):
    """
    Perform one-hot encoding on multiple categorical columns.
    """

    df_copy = df.copy()

    for col in columns:
        df_copy = one_hot_encode_column(df_copy, col, drop_first)

    return df_copy

def label_encode_column(df, column_name):
    """
    Perform label encoding on a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    column_name : str
        Column to encode

    Returns
    -------
    pd.DataFrame, LabelEncoder
        Encoded DataFrame and fitted encoder
    """

    df_copy = df.copy()

    encoder = LabelEncoder()
    df_copy[column_name + "_Encoded"] = encoder.fit_transform(df_copy[column_name].astype(str))

    return df_copy, encoder

def label_encode_multiple(df, columns):
    """
    Perform label encoding on multiple categorical columns.

    Returns encoded dataframe and dictionary of encoders.
    """

    df_copy = df.copy()
    encoders = {}

    for col in columns:
        encoder = LabelEncoder()
        df_copy[col + "_Encoded"] = encoder.fit_transform(df_copy[col].astype(str))
        encoders[col] = encoder

    return df_copy, encoders

#-----------------------------------------------
# Handling Missing Values
#-------------------------------------------  

def impute_missing_values(
    df,
    strategy="mean",
    columns=None,
    n_neighbors=5
):

    """
    Unified function to handle missing value imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    strategy : str
        Imputation strategy: "mean", "median", "mode", or "knn"
    columns : list or None
        Columns to impute. If None, auto-select numeric columns
    n_neighbors : int
        Used only for KNN imputation

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values
    """

    df_copy = df.copy()

    # Auto-select numeric columns if not provided
    if columns is None:
        columns = df_copy.select_dtypes(include="number").columns

    # Mean / Median / Mode Imputation
    if strategy in ["mean", "median", "mode"]:
        if strategy == "mode":
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            imputer = SimpleImputer(strategy=strategy)

        df_copy[columns] = imputer.fit_transform(df_copy[columns])

    # KNN Imputation
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_copy[columns] = imputer.fit_transform(df_copy[columns])

    else:
        raise ValueError("Strategy must be: 'mean', 'median', 'mode', or 'knn'")

    return df_copy


def remove_missing_values(df, axis=0, how="any", threshold=None, subset=None):
    """
    Remove rows or columns with missing values.
    """
    df_copy = df.copy()
    
    # If a threshold is provided, pandas requires 'how' to be None
    if threshold is not None:
        df_copy = df_copy.dropna(
            axis=axis,
            thresh=threshold,
            subset=subset
        )
    else:
        df_copy = df_copy.dropna(
            axis=axis,
            how=how,
            subset=subset
        )
        

def remove_missing_values(
    df,
    axis=0,
    how="any",
    threshold=None,
    subset=None
):
    """
    Remove rows or columns with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    axis : int
        0 = drop rows, 1 = drop columns
    how : str
        'any' -> drop if any missing
        'all' -> drop if all missing
    threshold : int or None
        Require that many non-NA values to keep row/column
    subset : list or None
        Columns to consider when dropping rows

    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """

    df_copy = df.copy()

    df_copy = df_copy.dropna(
        axis=axis,
        how=how,
        thresh=threshold,
        subset=subset
    )


    return df_copy

def drop_missing_rows(df, how="any", subset=None):
    """Drop rows with missing values."""
    return remove_missing_values(df, axis=0, how=how, subset=subset)

def drop_missing_columns(df, how="any", threshold=None):
    """Drop columns with missing values."""
    return remove_missing_values(df, axis=1, how=how, threshold=threshold)


#-----------------------------------------------
# Normalize/Standardize Numerical Features
#------------------------------------------- 

def normalize_minmax(df, columns=None):
    """
    Normalize selected columns to range [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    columns : list or None
        Columns to normalize. If None, use all numeric columns.

    Returns
    -------
    pd.DataFrame
        Normalized dataset
    """

    df_copy = df.copy()

    # Select numeric columns if not provided
    if columns is None:
        columns = df_copy.select_dtypes(include="number").columns

    scaler = MinMaxScaler()

    df_copy[columns] = scaler.fit_transform(df_copy[columns])

    return df_copy

 

def normalize_minmax(df, columns=None):
    """
    Normalize selected columns to range [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    columns : list or None
        Columns to standardize. If None, use all numeric columns.

        Columns to normalize. If None, use all numeric columns.

    Returns
    -------
    pd.DataFrame
        Standardized dataset

        Normalized dataset
    """

    df_copy = df.copy()

    # Auto-select numeric columns if not provided
    if columns is None:
        columns = df_copy.select_dtypes(include="number").columns

    scaler = StandardScaler()

    df_copy[columns] = scaler.fit_transform(df_copy[columns])

    return df_copy

#-----------------------------------------------
# Feature Engineering with WoE and IV
#------------------------------------------- 

def perform_woe_iv_feature_engineering(X_train,X_test, y_train):
    """
    Apply WoE transformation and calculate Information Value (IV).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataset
    X_test : pd.DataFrame
        Testing feature dataset
    y_train : pd.Series
        Training target variable

    Returns
    -------
    tuple
        (
            X_train_woe,
            X_test_woe,
            iv_table,
            fitted_woe_transformer
        )
    """

    # Initialize WoE transformer
    woe = WOE()

    # Fit on training data only
    woe.fit(X_train, y_train)

    # Transform datasets
    X_train_woe = woe.transform(X_train)
    X_test_woe = woe.transform(X_test)

    # Extract Information Value (IV)
    iv_table = (
        woe.iv_df.reset_index()
        .rename(
            columns={
                "index": "Feature",
                "Information_Value": "IV"
            }
        )
        .sort_values(
            by="IV",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        X_train_woe,
        X_test_woe,
        iv_table,
        woe
    )




