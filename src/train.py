import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X, y, path):
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, path)
    return model