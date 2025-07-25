import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

# Preprocessing pipelines
numeric_features_selector = make_column_selector(dtype_include='number')
categorical_features_selector = make_column_selector(dtype_include=['object', 'category'])
all_features_selector = make_column_selector(dtype_include=['number', 'object', 'category'])

simple_imputer_categorical = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='if_binary', handle_unknown="ignore"))
])


class RandomNumericImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.means_ = None  # Attribute to store the mean of each column
        self.std_ = None  # Attribute to store the standard deviations of each column
        self.columns = None

    def fit(self, X, y=None):
        """
        Compute the mean of each column and store it.

        Parameters:
        - X: Input data (numpy array or pandas DataFrame)
        - y: Ignored, exists for compatibility with sklearn pipelines

        Returns:
        - self: Fitted transformer
        """
        if isinstance(X, np.ndarray):
            mean = np.nanmean(X, axis=0)
            self.means_ = np.where(np.isnan(mean), 0, mean)

            std = np.std(X, axis=0)
            # exception in case std is calculated to nan
            self.std_ = np.where(np.isnan(std), 1, std)
        else:
            mean = X.mean(axis=0, skipna=True)
            self.means_ = mean.where(mean.notna(), 0)

            std = X.std(axis=0, skipna=True)
            # exception in case std is calculated to nan
            self.std_ = std.where(std.notna(), 1)
        return self

    def transform(self, X):
        """
        Replace missing values (np.nan) with the computed column means.

        Parameters:
        - X: Input data (numpy array or pandas DataFrame)

        Returns:
        - Transformed data (same type as input)
        """
        if isinstance(X, np.ndarray):
            random_X = np.array(
                [np.random.normal(self.means_[i], self.std_[i], len(X)) for i in range(len(self.means_))]
            )
            random_X = random_X.T
            X = np.where(np.isnan(X), random_X, X)
        else:
            gather_new_X = []
            for col in X:
                random_imputed_col = X[col].apply(lambda x: np.random.normal(self.means_[col], self.std_[col]))
                gather_new_X.append(X[col].fillna(random_imputed_col))
            X = pd.concat(gather_new_X, axis=1)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y).transform(X)


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='ID', limit=12, fill_medians=True):
        self.id_column = id_column
        self.limit = limit
        self.fill_medians = fill_medians
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # No fitting necessary for ffill
        self.feature_names_in_ = X.columns.to_list()
        self.medians = X.select_dtypes(include=["int", "float"]).median()

        return self

    def transform(self, X):
        X = X.copy()

        # Forward fill within each group up to the specified limit
        X = X.groupby(level=self.id_column).apply(
            lambda group: group.ffill(limit=self.limit)
        )

        X.index = X.index.droplevel(level=0)
        if self.fill_medians:
            X.loc[:, self.medians.index] = X.loc[:, self.medians.index].fillna(self.medians, axis=0)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        elif self.feature_names_in_ is not None:
            return self.feature_names_in_
        else:
            raise AttributeError("No feature names are available. "
                                 "Call fit before get_feature_names_out.")


class MeasurementCountImputer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='ID', count_column_prefix='n_measurements_'):
        self.id_column = id_column
        self.count_column_prefix = count_column_prefix
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_list()
        return self

    def transform(self, X):
        X = X.copy()

        # Create a column that counts the cumulative number of measurements per ID
        n_df = X.groupby(self.id_column).expanding().count().add_prefix(self.count_column_prefix).droplevel(level=0)
        self.new_col_names = n_df.columns
        return n_df

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return self.new_col_names
        elif self.feature_names_in_ is not None:
            return self.new_col_names
        else:
            raise AttributeError("No feature names are available. "
                                 "Call fit before get_feature_names_out.")


class MeasurementRollingCountImputer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='ID', n_periods=12, count_column_prefix='n_measurements_'):
        self.id_column = id_column
        self.count_column_prefix = count_column_prefix
        self.n_periods = n_periods
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_list()
        return self

    def transform(self, X):
        X = X.copy()

        # Create a column that counts the cumulative number of measurements per ID
        n_df = (
            X.groupby(self.id_column).rolling(window=self.n_periods, min_periods=1)
            .count().add_prefix(f"{self.count_column_prefix}{self.n_periods}_").droplevel(level=0)
        )
        self.new_col_names = n_df.columns
        return n_df

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return self.new_col_names
        elif self.feature_names_in_ is not None:
            return self.new_col_names
        else:
            raise AttributeError("No feature names are available. "
                                 "Call fit before get_feature_names_out.")


class RollingMeanSTDImputer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='ID', n_periods=12):
        self.id_column = id_column
        self.feature_names_in_ = None
        self.n_periods = n_periods

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_list()
        return self

    def transform(self, X):
        X = X.copy()

        mean_df = X.groupby(self.id_column).rolling(window=self.n_periods, min_periods=2).std().add_prefix("Mean_").droplevel(level=0).fillna(0)
        self.mean_col_names = mean_df.columns

        std_df = X.groupby(self.id_column).rolling(window=self.n_periods, min_periods=2).std().add_prefix("STD_").droplevel(level=0).fillna(0)
        self.std_col_names = std_df.columns
        return pd.concat([mean_df, std_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return self.mean_col_names.tolist() + self.std_col_names.tolist()
        elif self.feature_names_in_ is not None:
            return self.mean_col_names.tolist() + self.std_col_names.tolist()
        else:
            raise AttributeError("No feature names are available. "
                                 "Call fit before get_feature_names_out.")


class MeanSTDImputer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='ID'):
        self.id_column = id_column
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_list()
        return self

    def transform(self, X):
        X = X.copy()

        mean_df = X.groupby(self.id_column).expanding(min_periods=2).std().add_prefix("Mean_").droplevel(level=0).fillna(0)
        self.mean_col_names = mean_df.columns

        std_df = X.groupby(self.id_column).expanding(min_periods=2).std().add_prefix("STD_").droplevel(level=0).fillna(0)
        self.std_col_names = std_df.columns
        return pd.concat([mean_df, std_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return self.mean_col_names.tolist() + self.std_col_names.tolist()
        elif self.feature_names_in_ is not None:
            return self.mean_col_names.tolist() + self.std_col_names.tolist()
        else:
            raise AttributeError("No feature names are available. "
                                 "Call fit before get_feature_names_out.")

median_imputer = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
     numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

median_mean_std_imputer = ColumnTransformer([
    ('num_rolling', RollingMeanSTDImputer(), numeric_features_selector),
    ('num_expanding', MeanSTDImputer(), numeric_features_selector),
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
     numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

median_counter_imputer = ColumnTransformer([
    ('count_rolling', MeasurementRollingCountImputer(), numeric_features_selector),
    ('count_expanding', MeasurementCountImputer(), numeric_features_selector),
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
     numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

ffill_missing_imputer = ColumnTransformer([
    ("both", MissingIndicator(features="all"), all_features_selector),
    ("num", ForwardFillImputer(), numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

ffill_median_imputer = ColumnTransformer([
    ("num", ForwardFillImputer(), numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

median_imputer_missing = ColumnTransformer([
    ("both", MissingIndicator(features="all"), all_features_selector),
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
     numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

median_imputer_missing_robust = ColumnTransformer([
    ("both", MissingIndicator(features="all"), all_features_selector),
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]),
     numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

knn_imputer = ColumnTransformer([
    ('num', Pipeline([('imputer', KNNImputer()), ('scaler', StandardScaler())]), numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

only_missing_indicator = ColumnTransformer([
    ("both", MissingIndicator(features="all"), all_features_selector)
])

random_imputer = ColumnTransformer([
    ('num', Pipeline([("imputer", RandomNumericImputer()), ("scaler", StandardScaler())]), numeric_features_selector)
])

knn_imputer_missing = ColumnTransformer([
    ("both", MissingIndicator(features="all"), all_features_selector),
    ('num', Pipeline([('imputer', KNNImputer()), ('scaler', StandardScaler())]), numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

knn_imputer_missing_robust = ColumnTransformer([
    ("both", MissingIndicator(features="all"), all_features_selector),
    ('num', Pipeline([('imputer', KNNImputer()), ('scaler', RobustScaler())]), numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])

iterative_imputer = ColumnTransformer([
    ('num', Pipeline([('imputer', IterativeImputer()), ('scaler', StandardScaler())]), numeric_features_selector),
    ('cat', simple_imputer_categorical, categorical_features_selector)
])
