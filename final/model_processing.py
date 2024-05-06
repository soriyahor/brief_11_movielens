import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


partition_options = {"test_size": 0.2, "mini_size": 0.03}

normalize_options = {"min": 1, "max": 5}

model_options = {
    "n_components": 10,
    "max_iter": 200,
    "normalize": {"should": True, "min": 1, "max": 5},
}


def partition(df, opts=partition_options):
    """
    Partition the dataset into a training set and a test set

    :param df: Dataframe containing every rates of every user for every movie
    :type df: pandas.DataFrame

    :param opts: Options for the partition :
        - test_size: Size of the test set
        - mini_size: Size of the mini set

    :return: The training set, the test set, the mini training set and the mini test set

    example:
    train_data, test_data, train_mini, test_mini = partition(df, partition_options)
    """

    # Partition the dataset into a training set and a test set
    train_data, test_data = train_test_split(
        df, test_size=opts["test_size"], random_state=42
    )

    # Create a mini-set for better performance
    train_mini = train_data[: int(opts["mini_size"] * len(train_data))]
    test_mini = test_data[: int(opts["mini_size"] * len(test_data))]

    return train_data, test_data, train_mini, test_mini


def normalize(ranking_matrix, opts=normalize_options):
    """
    Normalize the ranking matrix

    Inputs:
    - ranking_matrix: the matrix to normalize

    Options:
    - min: minimum normalization value
    - max: maximum normalization value

    Output:
    - The normalized matrix

    Example:

    normalized_matrix = normalize(ranking_matrix, normalize_options)
    """

    # Create a MinMaxScaler
    scaler = MinMaxScaler(feature_range=(opts["min"], opts["max"]))

    # Normalize each row of the matrix (by using a double transposition)
    ranking_matrix = scaler.fit_transform(ranking_matrix.T).T
    return ranking_matrix


def run_model(df, opts=model_options):
    """
    Run the model

    Inputs:
    - df: every rates of every user for every movie

    Options:
    - n_components: Number of latent features
    - max_iter: Maximum number of iterations
    - normalize:
        - should: should the prediction matrix be normalized
        - min: minimum normalization value
        - max: maximum normalization value

    Output:
    - the resulting model
    - the prediction dataframe

    Example:

    model, pred_df = run_model(train_data, model_options)
    """

    # Pivot train dataframe to get a matrix of users and their ratings for movies
    matrix = df.pivot(index="user_id", columns="movie_id", values="rating")

    # Fill NaN values with
    matrix = matrix.fillna(0)

    # Sparse ratings train dataframe
    matrix_sparse = matrix.astype(pd.SparseDtype("float", 0))

    # Create the model
    model = NMF(n_components=opts["n_components"], max_iter=opts["max_iter"])

    # Fit the model to the user-item train matrix
    U = model.fit_transform(matrix_sparse)  # User matrix train
    M = model.components_  # Item matrix

    pred = np.dot(U, M)
    pred_matrix = pd.DataFrame(pred, columns=matrix.columns, index=matrix.index)

    if opts["normalize"]["should"]:
        # Normalize the prediction matrix
        pred_matrix = normalize(pred_matrix, opts=opts["normalize"])

    # Convert the prediction matrix to a dataframe
    pred_df = pd.DataFrame(pred_matrix).stack().reset_index()
    pred_df.columns = ["user_id", "movie_id", "predict"]  # Rename columns

    return model, pred_df
