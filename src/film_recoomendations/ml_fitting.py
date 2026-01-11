import numpy as np
import pandas as pd

def create_transformed_features(features_df:pd.DataFrame)-> tuple[pd.DataFrame, pd.Series]:
    data = features_df.copy()

    # Derived numeric features
    data["log_budget"] = np.log1p(data["budget"].fillna(0))
    data["log_revenue"] = np.log1p(data["revenue"].fillna(0))
    data["log_vote_count"] = np.log1p(data["vote_count"].fillna(0))
    data["genres_count"] = data["genres"].fillna("").apply(lambda s: 0 if s == "" else len(s.split("|")))
    data["countries_count"] = data["production_countries"].fillna("").apply(lambda s: 0 if s == "" else len(s.split("|")))

    # Fill missing for text fields
    for col in ["overview", "keywords", "people_text", "genres", "production_countries", "original_language"]:
        data[col] = data[col].fillna("")

    # Target
    y = data["Rating"].astype(float)

    return data, y