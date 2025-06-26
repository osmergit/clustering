import numpy as np
from sklearn.cluster import SpectralBiclustering
import matplotlib.pyplot as plt

# Matriz sintética (10 filas x 10 columnas)
data = np.random.rand(10, 10)

# Crear un patrón local: filas 2-4 y columnas 3-6 tienen valores altos
data[2:5, 3:7] += 2  # Bicluster artificial

# Aplicar Spectral Biclustering
model = SpectralBiclustering(n_clusters=2, random_state=42)
model.fit(data)

# Obtener asignaciones de filas y columnas
rows = model.row_labels_
cols = model.column_labels_
# Visualizar la matriz original y los biclusters
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Matriz Original")
plt.imshow(data, cmap="viridis")
plt.subplot(122)
plt.title("Biclusters Identificados")
plt.imshow(data[np.argsort(rows)][:, np.argsort(cols)], cmap="viridis")
plt.show()