import os
import pandas as pd
import numpy as np

# H-statistic for interaction effects
def calculate_h_statistic(model, X, feature_1, feature_2):
    X_base = X.copy()
    X_f1 = X.copy()
    X_f2 = X.copy()
    X_both = X.copy()

    X_f1[feature_1] = X[feature_1].mean()
    X_f2[feature_2] = X[feature_2].mean()
    X_both[feature_1] = X[feature_1].mean()
    X_both[feature_2] = X[feature_2].mean()

    # Predict with the model
    pred_base = model.predict_proba(X_base)[:, 1]
    pred_f1 = model.predict_proba(X_f1)[:, 1]
    pred_f2 = model.predict_proba(X_f2)[:, 1]
    pred_both = model.predict_proba(X_both)[:, 1]

    # Calculate H-statistic
    joint_effect = pred_both - pred_base
    individual_effects = (pred_f1 - pred_base) + (pred_f2 - pred_base)
    interaction_effect = joint_effect - individual_effects
    h_statistic = np.var(interaction_effect) / np.var(joint_effect)
    
    return h_statistic