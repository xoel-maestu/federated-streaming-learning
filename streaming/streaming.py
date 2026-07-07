# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "river>=0.22",
#     "scikit-learn>=1.5",
#     "numpy",
#     "matplotlib",
#     "rich",
#     "tqdm",
# ]
# ///
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from rich import print as print
from river import (
    compose,
    datasets,
    drift,
    feature_extraction,
    forest,
    metrics,
    preprocessing,
    tree,
)
from river.naive_bayes import GaussianNB as rvGaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as skGaussianNB
from tqdm import tqdm

print("\n\n--- Aprendizaje continuo con River ---\n\n")

# Dataset ---------------------------------------------------------------------
dataset = datasets.Elec2()

counts = Counter(y for _, y in dataset)

print("Distribución de las etiquetas:")
for c, count in counts.items():
    print(f"{c}: {count} ({count / sum(counts.values()):.5%})")

X, Y = zip(*dataset)
X = [list(x.values()) for x in X]
X, Y = np.array(X), np.array(Y)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Entrenamiento batch: GaussianNB de scikit-learn -----------------------------
print("\n\n# --- scikit-learn GaussianNB ---")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

pipe_hat = skGaussianNB()
y_pred = pipe_hat.fit(x_train, y_train).predict(x_test)

# print(f"Puntos mal clasificados del total de {x_test.shape[0]} puntos: {(y_test != y_pred).sum()}")
print("F1 score:", f1_score(y_test, y_pred))
print("Accuracy score:", accuracy_score(y_test, y_pred))

# Entrenamiento incremental: GaussianNB de river ------------------------------
print("\n# --- river GaussianNB ---")

pipe_hat = rvGaussianNB()
report = metrics.ClassificationReport()
accuracy = metrics.Accuracy()
f1 = metrics.F1()
counter = 0

for X, Y in dataset:
    y_pred = pipe_hat.predict_one(X)
    pipe_hat.learn_one(X, Y)

    if counter > 0:
        report.update(Y, y_pred)
        accuracy.update(Y, y_pred)
        f1.update(Y, y_pred)

    # if counter % 1500 == 0:
    #   print(f'{f1}')

    counter += 1

print("F1 score:", f1.get())
print("Accuracy score:", accuracy.get())

print(report)

# Comparación batch vs incremental --------------------------------------------

print(
    """
[cyan]
Se obtiene un mejor rendimiento empleando aprendizaje continuo respecto al \
aprendizaje tradicional.
Esto se puede deber a que los métodos de aprendizaje continuo están preparados \
para trabajar con datos con dependencia temporal, lo que puede no ser el caso \
con el modelo tradicional escogido.
"""
)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Integración de ADWIN para manejo de concept drift ---------------------------
print("\n\n# --- Integración de ADWIN ---")

model = rvGaussianNB()
model_raw = rvGaussianNB()

report = metrics.ClassificationReport()
report_raw = metrics.ClassificationReport()

pred = list()
pred_raw = list()

detector = drift.ADWIN(delta=0.004)  # Mayor sensibilidad a cambios frente a 0.002
drifts = list()

for i, (x, y) in enumerate(dataset):
    y_pred = model.predict_one(x)
    y_pred_raw = model_raw.predict_one(x)
    model.learn_one(x, y)
    model_raw.learn_one(x, y)

    pred.append(y_pred)
    pred_raw.append(y_pred_raw)

    if y_pred is not None:
        report.update(y, y_pred)
        report_raw.update(y, y_pred_raw)

    detector.update(x["nswprice"])
    if detector.drift_detected:
        print(
            f"Cambio detectado en las it.:[yellow]",
            ", ".join((str(i) for i in drifts)),
            end="\r",
        )  # En script funciona bien, lo mantengo por consistencia
        drifts.append(i)

        model = rvGaussianNB()
print()

print("Modelo con reinicio al detectar cambios:")
print(report)

print("Modelo en crudo:")
print(report_raw)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Modelos adaptativos: HAT y ARF ----------------------------------------------

pipe_hat = compose.Pipeline(
    preprocessing.StandardScaler(),
    feature_extraction.RBFSampler(),
    tree.HoeffdingAdaptiveTreeClassifier(),
)
pipe_arf = compose.Pipeline(
    preprocessing.StandardScaler(),
    feature_extraction.RBFSampler(),
    forest.ARFClassifier(),
)

report_hat = metrics.ClassificationReport()
report_arf = metrics.ClassificationReport()
acc_hat = metrics.Accuracy()
acc_arf = metrics.Accuracy()

accuracies = {"HAT": list(), "ARF": list()}

for x, y in tqdm(dataset):
    y_pred_1 = pipe_hat.predict_one(x)
    y_pred_2 = pipe_arf.predict_one(x)
    pipe_hat.learn_one(x, y)
    pipe_arf.learn_one(x, y)

    if y_pred_1 is not None:
        report_hat.update(y, y_pred_1)
        acc_hat.update(y, y_pred_1)

        accuracies["HAT"].append(acc_hat.get())

    if y_pred_2 is not None:
        report_arf.update(y, y_pred_2)
        acc_arf.update(y, y_pred_2)

        accuracies["ARF"].append(acc_arf.get())
print()

print("HAT")
print(report_hat)
print("\nARF")
print(report_arf)

plt.figure(figsize=(16, 8))
plt.plot(accuracies["HAT"], label="HAT")
plt.plot(accuracies["ARF"], label="ARF")
plt.xlabel("Número de muestras")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Evolución del accuracy de modelos adaptativos")
plt.savefig("adaptive_models_accuracy.png", dpi=150, bbox_inches="tight")
print("Figura guardada en adaptive_models_accuracy.png")
plt.show()
