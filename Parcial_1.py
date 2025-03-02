import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Función para leer datos desde un archivo txt con formato "x y"
def ReadData(filename):
    x_data, y_data = [], []
    with open(filename, "r") as file:
        for line in file:
            x_val, y_val = map(float, line.split())  # Separa valores por espacios
            x_data.append(x_val)
            y_data.append(y_val)
    return np.array(x_data), np.array(y_data)

# Leer datos desde el archivo
x, y = ReadData("Datos_Taller_2_B.txt")  # Archivo con valores de x e y en columnas

def funcion_modelo(x, a0, a1, a2, modelo):
    if modelo == 1:
        return a0 - a1 * np.exp(-a2 * x)
    elif modelo == 2:
        return a0 * x - a1 * np.exp(-a2 * x)
    elif modelo == 3:
        return a0 * x**2 - a1 * np.exp(-a2 * x)

def jacobiano(x, a0, a1, a2, modelo):
    J = np.zeros((len(x), 3))
    if modelo == 1:
        J[:, 0] = 1  # Derivada respecto a a0
        J[:, 1] = -np.exp(-a2 * x)  # Derivada respecto a a1
        J[:, 2] = a1 * x * np.exp(-a2 * x)  # Derivada respecto a a2
    elif modelo == 2:
        J[:, 0] = x  # Derivada respecto a a0
        J[:, 1] = -np.exp(-a2 * x)  # Derivada respecto a a1
        J[:, 2] = a1 * x * np.exp(-a2 * x)  # Derivada respecto a a2
    elif modelo == 3:
        J[:, 0] = x**2  # Derivada respecto a a0
        J[:, 1] = -np.exp(-a2 * x)  # Derivada respecto a a1
        J[:, 2] = a1 * x * np.exp(-a2 * x)  # Derivada respecto a a2
    return J

def ajuste_regresion_no_lineal(x, y, modelo, a0_init=1, a1_init=1, a2_init=1, error_tol=0.01, max_iter=100):
    a0, a1, a2 = a0_init, a1_init, a2_init
    for _ in range(max_iter):
        y_pred = funcion_modelo(x, a0, a1, a2, modelo)
        error = y - y_pred
        J = jacobiano(x, a0, a1, a2, modelo)
        delta_a, _, _, _ = np.linalg.lstsq(J, error, rcond=None)
        a0 += delta_a[0]
        a1 += delta_a[1]
        a2 += delta_a[2]
        if np.linalg.norm(delta_a) < error_tol:
            break
    return a0, a1, a2


# Ajuste y graficación de los tres modelos
modelos = [1, 2, 3]
plt.figure(figsize=(12, 4))


for i, modelo in enumerate(modelos, 1):
    a0, a1, a2 = ajuste_regresion_no_lineal(x, y, modelo)
    print(f'Modelo {modelo}: a0 = {a0:.4f}, a1 = {a1:.4f}, a2 = {a2:.4f}')
    
    plt.figure()
    plt.scatter(x, y, label='Datos experimentales')
    x_smooth = np.linspace(min(x), max(x), 100)
    plt.plot(x_smooth, funcion_modelo(x_smooth, a0, a1, a2, modelo), 'r-', label=f'Modelo {modelo}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Modelo {modelo}')

plt.tight_layout()
plt.show()
