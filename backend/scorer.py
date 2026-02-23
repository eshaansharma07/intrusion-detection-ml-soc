from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res