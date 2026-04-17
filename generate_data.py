import numpy as np
import pandas as pd

np.random.seed(42)

n = 1200

feature1 = np.random.normal(0, 1, n)
feature2 = np.random.normal(0, 1, n)
feature3 = 0.5 * feature1 + np.random.normal(0, 0.5, n)
feature4 = np.sin(np.linspace(0, 12, n)) + np.random.normal(0, 0.2, n)

# regression target
target_reg = (
    1.5 * feature1
    - 0.8 * feature2
    + 0.6 * feature3
    + 0.3 * feature4
    + np.random.normal(0, 0.4, n)
)

# classification target
target_clf = (target_reg > np.median(target_reg)).astype(int)

df = pd.DataFrame({
    "feature1": feature1,
    "feature2": feature2,
    "feature3": feature3,
    "feature4": feature4,
    "target_reg": target_reg,
    "target_clf": target_clf,
})

df.to_csv("data/sample.csv", index=False)
print(f"Saved {len(df)} rows to data/sample.csv")