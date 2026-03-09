import pickle
from pathlib import Path
import pandas as pd
from typing import Any, Dict, Optional


MODEL_PATH = Path("linear_regression_model.pkl")


def _load_metrics(path: str = "linear_regression_metrics.pkl") -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Не найден файл linear_regression_model.pkl. Сначала запустите linear_regression.py, "
        "чтобы обучить и сохранить модель."
    )

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

metrics = _load_metrics()

features = [
    ("alcohol",              "Алкоголь (%)"),
    ("fixed acidity",        "Фиксированная кислотность"),
    ("residual sugar",       "Остаточный сахар (г/л)"),
    ("citric acid",          "Лимонная кислота (г/л)"),
    ("pH",                   "pH"),
    ("chlorides",            "Хлориды (г/л)"),
    ("sulphates",            "Сульфаты (г/л)"),
    ("volatile acidity",     "Летучая кислотность (г/л)"),
    ("free sulfur dioxide",  "Свободный SO₂ (мг/л)"),
    ("total sulfur dioxide", "Общий SO₂ (мг/л)"),
]

print("=" * 45)
print("   Предсказание плотности красного вина")
print("=" * 45)

if metrics is not None:
    r2 = metrics.get("r2")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    if r2 is not None:
        print(f"  Точность (R² на тесте): {float(r2):.4f} ({float(r2) * 100:.2f}%)")
    if mae is not None:
        print(f"  Ошибка (MAE на тесте): {float(mae):.6f}")
    if rmse is not None:
        print(f"  Ошибка (RMSE на тесте): {float(rmse):.6f}")
    print("-" * 45)

values = []
feature_names = [name for name, _ in features]
for _, label in features:
    while True:
        raw = input(f"  {label}: ").strip()
        try:
            values.append(float(raw))
            break
        except ValueError:
            print("    [!] Введите числовое значение.")

if metrics is not None and isinstance(metrics.get("feature_names"), list) and metrics["feature_names"]:
    feature_names = metrics["feature_names"]

X_input = pd.DataFrame([values], columns=feature_names)
prediction = model.predict(X_input)[0]

print("=" * 45)
print(f"  Предсказанная плотность: {prediction:.6f} г/см³")

raw_true = input("  Фактическая плотность (Enter чтобы пропустить): ").strip()
if raw_true:
    try:
        true_density = float(raw_true)
        abs_err = abs(true_density - float(prediction))
        print(f"  Абсолютная ошибка: {abs_err:.6f} г/см³")
    except ValueError:
        print("  [!] Фактическая плотность должна быть числом (пропускаю).")

print("=" * 45)

