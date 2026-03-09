import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("winequality-red.csv")

X = df[["alcohol", "fixed acidity", "residual sugar", "citric acid", "pH",
        "chlorides", "sulphates", "volatile acidity", "free sulfur dioxide", "total sulfur dioxide"]]
Y = df["density"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

r2 = r2_score(Y_test, Y_pred)
print(f"R²: {r2:.4f} ({r2*100:.2f}%)")

x_min, x_max = Y_test.min(), Y_test.max()
y_min, y_max = Y_pred.min(), Y_pred.max() 

plt.figure(figsize=(7, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5, color="steelblue", label="Тестовые точки")
plt.plot([x_min, x_max], [y_min, y_max], color="tomato", linewidth=2, label="Идеальное предсказание")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title(f"Y_test vs Y_pred  |  R² = {r2*100:.2f}%")
plt.xlabel("Y_test")
plt.ylabel("Y_pred")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_plot.png", dpi=150)
print("График сохранён: linear_regression_plot.png")