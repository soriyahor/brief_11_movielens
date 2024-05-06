import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score


def rate(
    predict_df,
    cleaned_df,
    opt_mse=True,
    opt_top_10=True,
    opt_bottom_10=True,
    opt_ndcg=True,
):
    """
    Create indicators on the Movielens DB.

    Inputs:
    - the prediction DF
    - the cleaned rating DF

    Options:
    - mse indicator : the Mean Squared Error
    - top_10 : compare top 10 movies form the prediction to the user ratings (5)
    - bottom_10 : compare worse 10 movies form the prediction to the user ratings (2)
    - ndcg : Compute Normalized Discounted Cumulative Gain.
    """
    compare_df = pd.merge(
        predict_df, cleaned_df, how="inner", on=["user_id", "movie_id"]
    )
    sorted_df = compare_df.sort_values(
        by=["user_id", "predict"], ascending=[True, False]
    ).groupby("user_id")

    db = []

    if opt_mse:
        mse = mean_squared_error(compare_df["rating"], compare_df["predict"])
        # delta_mse = np.sqrt(mse)
        db.append(mse)

    if opt_top_10:
        average_top_rating = (sorted_df.head(10))["rating"].mean()
        diff_top_rating_5 = 5 - average_top_rating
        db.append(diff_top_rating_5)

    if opt_bottom_10:
        average_worse_rating = (sorted_df.tail(10))["rating"].mean()
        diff_worse_rating_2 = average_worse_rating - 2
        db.append(diff_worse_rating_2)

    if opt_ndcg:
        ndcg = ndcg_score([compare_df["rating"].values], [compare_df["predict"].values])
        db.append(ndcg)

    return db
