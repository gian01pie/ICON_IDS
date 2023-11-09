import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_class_distribution(df, column, title, xlabel, ylabel, figsize=(12, 7), color_map='jet', rotation=45):
    """
    Plots a class distribution histogram.

    Parameters:
    - df: pandas.DataFrame, the dataframe containing the data.
    - column: str, the name of the column to plot the distribution for.
    - title: str, the title of the plot.
    - xlabel: str, the label for the x-axis.
    - ylabel: str, the label for the y-axis.
    - figsize: tuple, the size of the figure (default is (12, 7)).
    - color_map: str, the matplotlib colormap to use for the bars (default is 'jet').
    - rotation: int, the degree of rotation for the x-axis tick labels (default is 45).
    """

    class_distribution = df[column].value_counts()
    total_observations = class_distribution.sum()
    percentages = (class_distribution.values / total_observations) * 100

    plt.figure(figsize=figsize)
    colors = plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(class_distribution)))
    bars = plt.bar(class_distribution.index, class_distribution.values, color=colors)

    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.03 * total_observations, f'{height}', ha='center',
                 va='bottom')
        plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{percentage:.2f}%', ha='center', va='center',
                 color='white')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_pie_chart_distribution(df, column, title, figsize=(10, 8), explode_threshold=0.05):
    """
    Plots a pie chart of a class distribution.

    Parameters:
    - df: pandas.DataFrame, the dataframe containing the data.
    - column: str, the name of the column to plot the distribution for.
    - title: str, the title of the plot.
    - figsize: tuple, the size of the figure (default is (10, 8)).
    - explode_threshold: float, the threshold percentage to apply the explode effect on slices (default is 0.05).
    """
    class_distribution = df[column].value_counts()
    total = class_distribution.sum()

    # Calculate explode values within a list comprehension
    explode_values = [0.1 if count < (total * explode_threshold) else 0 for count in class_distribution]

    # Create the pie chart
    plt.figure(figsize=figsize)
    plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=140,
            explode=explode_values)
    plt.title(title)
    plt.show()


def plot_grouped_bar(df, category, sub_category, count, title, xlabel, ylabel, figsize=(15, 7), rotation=45):
    """
    Plots a grouped bar chart with Seaborn.

    Parameters:
    - df: pandas.DataFrame, the dataframe containing the data.
    - category: str, the name of the main category column.
    - sub_category: str, the name of the sub-category column.
    - count: str, the name of the column with count data.
    - title: str, the title of the plot.
    - xlabel: str, the label for the x-axis.
    - ylabel: str, the label for the y-axis.
    - figsize: tuple, the size of the figure (default is (15, 7)).
    - rotation: int, the degree of rotation for the x-axis tick labels (default is 45).
    """
    grouped_data = df.groupby([category, sub_category]).size().reset_index(name=count)
    grouped_data = grouped_data.sort_values(by=count, ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(x=category, y=count, hue=sub_category, data=grouped_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.legend(title=sub_category, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def create_class_distribution_table(df, column_name):
    """
    Creates a table of class distributions for a given DataFrame and column.

    Parameters:
    - df: pandas.DataFrame, the dataframe containing the data.
    - column_name: str, the name of the column to analyze the distribution of.

    Returns:
    - A pandas DataFrame containing the class distribution table.
    """
    # Calculate value counts and percentages
    class_distribution = df[column_name].value_counts().sort_values(ascending=False)
    percentages = np.round(class_distribution / len(df) * 100, 3)

    # Create a DataFrame
    table = pd.DataFrame({
        'Attacco': class_distribution.index,
        'Numero di Esempi': class_distribution.values,
        'Percentuale': percentages
    })

    # Reset index to start from 1
    table.index = np.arange(1, len(table) + 1)

    # Return the table
    return table


def find_unique_values(df1, df2, column_name, df1_name='First Dataset', df2_name='Second Dataset'):
    """
    Finds unique values that are present in one dataframe column and not the other.

    Parameters:
    - df1: pandas.DataFrame, the first dataframe to compare.
    - df2: pandas.DataFrame, the second dataframe to compare.
    - column_name: str, the column name to compare for unique values.
    - df1_name: str, a human-readable name for the first dataframe.
    - df2_name: str, a human-readable name for the second dataframe.

    Returns:
    - A tuple of pandas.DataFrames: the first dataframe contains unique values in df1's column,
      the second dataframe contains unique values in df2's column.
    """
    # Set of unique values in each dataframe
    unique_values_df1 = set(df1[column_name].unique())
    unique_values_df2 = set(df2[column_name].unique())

    # Find the differences between the two sets
    values_only_in_df1 = unique_values_df1 - unique_values_df2
    values_only_in_df2 = unique_values_df2 - unique_values_df1

    # Create DataFrames for exclusive values
    df1_exclusive_values = pd.DataFrame(list(values_only_in_df1), columns=[f'Valori unici in {df1_name}'])
    df2_exclusive_values = pd.DataFrame(list(values_only_in_df2), columns=[f'Valori unici in {df2_name}'])

    # Return the DataFrames
    return df1_exclusive_values, df2_exclusive_values


def plot_line_distribution(df, column, figsize=(15,7), rotation=90, grid=True, title=None, xlabel=None, ylabel='Numero di Osservazioni'):
    """
    Plots a line distribution for a categorical feature.

    Parameters:
    - df: pandas.DataFrame, the dataframe containing the data.
    - column: str, the column name for which the distribution is to be plotted.
    - figsize: tuple, the size of the figure (width, height).
    - rotation: int, the rotation angle of x-tick labels.
    - grid: bool, whether to show grid lines.
    - title: str, the title of the plot.
    - xlabel: str, the label for the x-axis.
    - ylabel: str, the label for the y-axis.
    """
    # Calculate the value counts and reset index to convert to DataFrame suitable for sns.lineplot
    distribution = df[column].value_counts().reset_index()
    distribution.columns = [column, 'Count']

    plt.figure(figsize=figsize)
    sns.lineplot(x=column, y='Count', data=distribution, marker='o')
    plt.xticks(rotation=rotation)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
