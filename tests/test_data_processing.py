import pandas as pd
import pytest

def add_transaction_count(df, customer_col):
    return df.groupby(customer_col).size().reset_index(name="Transaction_Count")


def add_total_amount(df, customer_col, amount_col):
    return df.groupby(customer_col)[amount_col].sum().reset_index(name="Total_Amount")

def test_add_transaction_count_columns():
    """
    Test that transaction count function returns correct column.
    """

    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B", "B", "B"],
        "Amount": [100, 200, 300, 400, 500]
    })

    result = add_transaction_count(df, "CustomerId")

    # Check output column exists
    assert "Transaction_Count" in result.columns

    # Check correct number of rows
    assert len(result) == 2

def test_add_total_amount_values():
    """
    Test that total amount per customer is calculated correctly.
    """

    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B"],
        "Amount": [100, 200, 300]
    })

    result = add_total_amount(df, "CustomerId", "Amount")

    # Convert to dictionary for easy checking
    result_dict = dict(zip(result["CustomerId"], result["Total_Amount"]))

    # Expected values
    assert result_dict["A"] == 300
    assert result_dict["B"] == 300

def test_output_not_empty():
    """
    Ensure function does not return empty dataframe.
    """

    df = pd.DataFrame({
        "CustomerId": ["A", "B"],
        "Amount": [100, 200]
    })

    result = add_transaction_count(df, "CustomerId")

    assert not result.empty