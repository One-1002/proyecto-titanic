import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cargar el dataset train.csv
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "../data/train.csv")

df = pd.read_csv(csv_path)

# 2. Limpieza y preprocesamiento básico
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Variables nuevas requeridas
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


def categorizar_edad(edad):
  if edad <= 12:
    return "Niño"
  elif edad <= 25:
    return "Joven"
  elif edad <= 60:
    return "Adulto"
  else:
    return "Adulto mayor"


df["AgeCategory"] = df["Age"].apply(categorizar_edad)

print("=" * 50)
print("ANÁLISIS ESTADÍSTICOS Y PREGUNTAS CLAVE")
print("=" * 50)

# Análisis 1: Porcentaje total de pasajeros que sobrevivió
survival_rate = df["Survived"].mean() * 100
print(
    f"1. Porcentaje de supervivencia general: {survival_rate:.2f}% de los"
    " pasajeros."
)

# Análisis 2: Supervivencia según género
survival_by_sex = df.groupby("Sex")["Survived"].mean() * 100
print("\n2. Porcentaje de supervivencia según género:")
print(survival_by_sex.round(2))

# Análisis 3: Supervivencia según la clase del pasajero (Pclass)
survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100
print("\n3. Porcentaje de supervivencia según la clase del pasajero:")
print(survival_by_class.round(2))

# Análisis 4: Supervivencia según categoría de edad
survival_by_age_cat = df.groupby("AgeCategory")["Survived"].mean() * 100
print("\n4. Porcentaje de supervivencia según grupo de edad:")
print(survival_by_age_cat.round(2))


print("\n" + "=" * 50)
print("GENERACIÓN DE VISUALIZACIONES")
print("=" * 50)

# Crear directorio de salidas si no existe
output_dir = os.path.join(current_dir, "../outputs/resultados")
os.makedirs(output_dir, exist_ok=True)

# Configuración de estilo
sns.set_theme(style="whitegrid")

# Visualización 1: Tasa de supervivencia por Clase y Género
plt.figure(figsize=(8, 5))
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, palette="muted")
plt.title("Tasa de Supervivencia por Clase y Género")
plt.xlabel("Clase del Pasajero (Pclass)")
plt.ylabel("Tasa de Supervivencia")
plt.ylim(0, 1)
plt.legend(title="Género")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "supervivencia_clase_genero.png"))
plt.close()
print("-> Generada y guardada: supervivencia_clase_genero.png")

# Visualización 2: Distribución de Supervivencia por Categoría de Edad
plt.figure(figsize=(8, 5))
sns.barplot(
    x="AgeCategory",
    y="Survived",
    data=df,
    order=["Niño", "Joven", "Adulto", "Adulto mayor"],
    palette="Blues_d",
)
plt.title("Tasa de Supervivencia por Grupo de Edad")
plt.xlabel("Categoría de Edad")
plt.ylabel("Tasa de Supervivencia")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "supervivencia_edad.png"))
plt.close()
print("-> Generada y guardada: supervivencia_edad.png")

# Visualización 3: Relación entre la Tarifa (Fare) y la Supervivencia
plt.figure(figsize=(8, 5))
sns.boxplot(x="Survived", y="Fare", data=df, palette="Set2")
plt.title("Distribución de la Tarifa Pagada según Supervivencia")
plt.xlabel("Sobrevivió (0 = No, 1 = Sí)")
plt.ylabel("Tarifa Pagada (Fare)")
# Limitamos el eje Y para mejor visualización debido a valores atípicos extremos
plt.ylim(0, 300)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tarifa_vs_supervivencia.png"))
plt.close()
print("-> Generada y guardada: tarifa_vs_supervivencia.png")

print(
    "\n¡Todas las visualizaciones se han guardado exitosamente en"
    " outputs/resultados/!"
)