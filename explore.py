## Imports:
import numpy as np
import pandas as pd
from scipy import stats
from pydataset import data
import seaborn as sns
import matplotlib.pyplot as plt 
import acquire as acq
import prepare as prep
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
alpha=0.05
from sklearn.feature_selection import SelectKBest, f_classif
import random

## FUNCTIONS:
# BAR PLOT
def create_barplot(data, x, y, title):
    # convert y column to numeric data type
    data[y] = pd.to_numeric(data[y])
    
    # create barplot
    sns.barplot(data=data, x=x, y=y)
    plt.title(title)
    
    # plot average line
    plt.axhline(data[y].mean(), color='red', linestyle='--', label='Average')
    plt.legend()

# CHI SQUARE

def chi_square_test(observed, alpha=0.05):
    "this function will calculate the chi square and print out results"
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print("Observed Contingency Table:")
    print(observed)
    print("Expected Contingency Table:")
    print(expected)
    print("Chi-Square Test Statistic:")
    print(chi2)
    print("p-value:")
    print(p)
    if p < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')
    
# T-TEST:
def t_test(sample1, sample2):
    """
    Performs a two-sample t-test on the provided samples and returns the t-value and p-value.

    Parameters:
    sample1 (array-like): First sample data
    sample2 (array-like): Second sample data

    Returns:
    t_value (float): The calculated t-value from the t-test
    p_value (float): The calculated p-value from the t-test
    """
    # Convert samples to numpy arrays
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    # Calculate t-value and p-value using scipy's ttest_ind function
    t_value, p_value = ttest_ind(sample1, sample2)

    return t_value, p_value

# DECISION TREE:

def fit_DT_random_features(df, target_col, n_random_features=0):
    """
    Fits a decision tree model to the input DataFrame `df`, with an additional
    `n_random_features` randomly selected features from the same DataFrame.
    Returns the best model that achieves a score of 79% or higher on the
    specified `target_col` variable.
    """
    X = df.select_dtypes(exclude=['object']).drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    best_score = 0
    best_model = None
    for depth in range(1, 21):
        selected_cols = np.random.choice(X.columns, size=n_random_features, replace=False)
        X_train_with_random = X_train.copy()
        X_test_with_random = X_test.copy()
        for col in selected_cols:
            X_train_with_random[col] = np.random.permutation(X_train_with_random[col].values)
            X_test_with_random[col] = np.random.permutation(X_test_with_random[col].values)
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train_with_random, y_train)
        score = model.score(X_test_with_random, y_test)
        if score > best_score and score >= 0.79:
            best_score = score
            best_model = model
    return best_model

# KNN:
def fit_knn_random_features(df, target_col, n_random_features=0):
    X = df.select_dtypes(exclude=['object']).drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_score = 0
    best_model = None
    for k in range(1, 21):
        selected_cols = np.random.choice(X.columns, size=n_random_features, replace=False)
        X_train_with_random = X_train.copy()
        X_test_with_random = X_test.copy()
        for col in selected_cols:
            X_train_with_random[col] = np.random.permutation(X_train_with_random[col].values)
            X_test_with_random[col] = np.random.permutation(X_test_with_random[col].values)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_with_random, y_train)
        score = model.score(X_test_with_random, y_test)
        if score > best_score and score >= 0.79:
            best_score = score
            best_model = model
    return best_model

#PIE CHART

def create_pie_chart(df, column_name,title):
    values = df[column_name].value_counts()
    labels = values.index.tolist()
    sizes = values.tolist()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()

# KNN CHART


def plot_knn_test_performance(knn_model, x_test, y_test, plot_type='bar'):
    "Make predictions on the test data"
    y_pred = knn_model.predict(x_test)

    "Create a plot of the predicted versus actual values"
    if plot_type == 'bar':
        unique_y_test = np.unique(y_test)
        count_y_test = [np.sum(y_test == u) for u in unique_y_test]
        count_y_pred = [np.sum(y_pred == u) for u in unique_y_test]
        plt.bar(unique_y_test, count_y_test, color='blue', alpha=0.5, label='Actual')
        plt.bar(unique_y_test, count_y_pred, color='red', alpha=0.5, label='Predicted')
        'Set the x-axis and y-axis labels'
        plt.xlabel('Class')
        plt.ylabel('Count')
        ' Set the plot title'
        plt.title('KNN Test Performance')
        'Add legend'
        plt.legend()
    elif plot_type == 'pie':
        unique_y_test, count_y_test = np.unique(y_test, return_counts=True)
        count_y_pred = [np.sum(y_pred == u) for u in unique_y_test]
        pastel_colors = plt.cm.Set3(np.linspace(0, 0.5, len(unique_y_test)))
        plt.pie(count_y_test, labels=unique_y_test, colors=pastel_colors, 
                autopct='%1.1f%%', startangle=90, counterclock=False, labeldistance=1.05, wedgeprops=dict(width=0.5))
        plt.pie(count_y_pred, colors=pastel_colors, 
                autopct='%1.1f%%', startangle=90, counterclock=False, labeldistance=0.8, wedgeprops=dict(width=0.3, edgecolor='w'))
        ' Set the plot title'
        plt.title('KNN Test Performance\nActual vs Predicted', y=0.8)
    else:
        raise ValueError('Invalid plot type specified. Please choose either "bar" or "pie".')

    ' Show the plot'
    plt.show()




















    


    
