import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np
%matplotlib inline
import random

rng = np.random.RandomState(1)

x = 37 * rng.rand(50) + 1 # 1度から3度
y = 500 * x - 5000 * rng.randn(50) # 売り上げデータはおよそ気温の500倍想定、ただしノイズを組みこむ

# 線形回帰
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression(fit_intercept=True)

model_lr.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 50, 1000)
yfit_lr = model.predict(xfit[:, np.newaxis])

# リッジ回帰
from sklearn.linear_model import Ridge
model_r = Ridge(alpha=1000)
model_r.fit(x[:, np.newaxis], y)

yfit_r = model_r.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit_lr,color="red", label="LinearRegression");
plt.plot(xfit, yfit_r, color="yellow", label="Ridge alpha=1000")
plt.legend()

print("linear regression: slope ", model_lr.coef_[0], "interception", model_lr.intercept_)
print("ridge regression: slope ", model_r.coef_[0], "interception", model_r.intercept_)
