# Parameter grids
rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__class_weight': ["balanced", None]
}

xgb_param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__scale_pos_weight': [1, 25, 99]
}