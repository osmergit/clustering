# -*- coding: utf-8 -*-
"""
Ejemplo de Calibración de Modelos de Clasificación:
1. Comparación entre modelo no calibrado (Random Forest) y calibrado (Platt Scaling).
2. Visualización de curvas de calibración.
3. Métricas de evaluación (Brier Score).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8')
# =============================================
# 1. Generar datos y entrenar modelo inicial
# =============================================
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10, 
    random_state=42
)
print(X.shape)
print(X)
print(y.shape)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("los datos del entrenamiento para el modelo")
print(X_train.shape)
print("los datos de prueba para el modelo")
print(X_test.shape)
# Modelo sin calibrar (Random Forest tiende a estar mal calibrado)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
prob_uncalibrated = model.predict_proba(X_test)[:, 1]  # Probabilidades predichas

# =============================================
# 2. Calibrar el modelo con Platt Scaling
# =============================================
calibrated_model = CalibratedClassifierCV(
    model, 
    method='sigmoid',  # Platt Scaling
    cv=3  # Validación cruzada
)
calibrated_model.fit(X_train, y_train)
prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# =============================================
# 3. Evaluación y gráficos
# =============================================
# Curvas de calibración
fop_uncalibrated, mpv_uncalibrated = calibration_curve(y_test, prob_uncalibrated, n_bins=10)
fop_calibrated, mpv_calibrated = calibration_curve(y_test, prob_calibrated, n_bins=10)

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(mpv_uncalibrated, fop_uncalibrated, 's-', label="Random Forest (Sin calibrar)")
plt.plot(mpv_calibrated, fop_calibrated, 'o-', label="Platt Scaling (Calibrado)")
plt.plot([0, 1], [0, 1], 'k--', label="Perfectamente calibrado")
plt.xlabel("Probabilidad predicha", fontsize=12)
plt.ylabel("Fracción de positivos reales", fontsize=12)
plt.title("Curva de Calibración", fontsize=15)
plt.legend()
plt.grid(True)
plt.show()

# =============================================
# 4. Métricas de evaluación (Brier Score)
# =============================================
brier_uncalibrated = brier_score_loss(y_test, prob_uncalibrated)
brier_calibrated = brier_score_loss(y_test, prob_calibrated)

print("\n=== Métricas de Calibración ===")
print(f"Brier Score (Sin calibrar): {brier_uncalibrated:.4f}")
print(f"Brier Score (Calibrado): {brier_calibrated:.4f}")
print("(Menor es mejor)")

# =============================================
# 5. Comparación con Isotonic Regression (Opcional)
# =============================================
calibrated_isotonic = CalibratedClassifierCV(
    model, 
    method='isotonic',  # Regresión isotónica
    cv=3
)
calibrated_isotonic.fit(X_train, y_train)
prob_isotonic = calibrated_isotonic.predict_proba(X_test)[:, 1]

fop_isotonic, mpv_isotonic = calibration_curve(y_test, prob_isotonic, n_bins=10)
brier_isotonic = brier_score_loss(y_test, prob_isotonic)
# Gráfico comparativo
plt.figure(figsize=(10, 6))
plt.plot(mpv_uncalibrated, fop_uncalibrated, 's-', label="Sin calibrar")
plt.plot(mpv_calibrated, fop_calibrated, 'o-', label="Platt Scaling")
plt.plot(mpv_isotonic, fop_isotonic, '^-', label="Isotonic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="Perfecto")
plt.xlabel("Probabilidad predicha", fontsize=12)
plt.ylabel("Fracción de positivos reales", fontsize=12)
plt.title("Comparación de Métodos de Calibración", fontsize=15)
plt.legend()
plt.grid(True)
plt.show()
print(f"\nBrier Score (Isotonic): {brier_isotonic:.4f}")