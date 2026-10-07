import pandas as pd                                   
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, export_text 

# LEER LA TABLA
tabla = pd.read_csv("TablaPrediccionAbandono.csv", sep=";", decimal=",", encoding="latin1")
print("Cantidad de alumnos (filas) y de columnas:", tabla.shape)

# PREPARAR LOS DATOS

# Solo se guardan las filas donde el resultado es "Abandono o Continua"
tabla = tabla[tabla["EstadoFinal"].isin(["Abandonó", "Continúa"])]

# Algunas Tecnicaturas estan con tilde, se normaliza
tabla["carrera"] = tabla["carrera"].str.replace("TÉCNICATURA", "TECNICATURA")

# Predice 0 Continua 1 Abandona
resultado = (tabla["EstadoFinal"] == "Abandonó").astype(int)

# Datos con los que se predice: todas las columnas menos "EstadoFinal"
datos = tabla.drop(columns=["EstadoFinal"])

# TEXTO A NUMERO
datos["genero"] = datos["genero"].map({"f": 0, "m": 1})
datos["trabaja/NoTrabaja"] = datos["trabaja/NoTrabaja"].map({"No": 0, "Si": 1})
datos["ActividadesExtracurriculares(Estudio)"] = datos["ActividadesExtracurriculares(Estudio)"].map({"No": 0, "Si": 1})

# Se crea una columna por carrera y se suma 1 si el alumno cursa esa carrera
datos = pd.get_dummies(datos, columns=["carrera"], dtype=int)


# ==============================================================
# ENTRENAR EL ARBOL
# ==============================================================
datos_entrenamiento, datos_prueba, resultado_entrenamiento, resultado_prueba = train_test_split(
    datos, resultado, test_size=0.25, random_state=42)

arbol = DecisionTreeClassifier(
    max_depth=3,            # Limita la altura del arbol (maximo 3 preguntas)
    min_samples_leaf=5,     # Exige que cada rama tenga al menos 5 alumnos
    random_state=42, 
    class_weight='balanced'
)

# El arbol aprende
arbol.fit(datos_entrenamiento, resultado_entrenamiento)       

# ==============================================================
# PASO 4: QUE TAN BIEN FUNCIONA 
# ==============================================================
precision = arbol.score(datos_prueba, resultado_prueba)
print("\n=== PRECISION DEL ARBOL ===")
print("Acierta en el", round(precision * 100, 1), "% de los alumnos de prueba")

# Comparacion: % que obtendriamos si predijeramos la opcion mas frecuente:
porcentaje_abandonan = resultado.mean()
mas_comun = max(porcentaje_abandonan, 1 - porcentaje_abandonan)
print("Acierto de referencia (opcion mas frecuente)", round(mas_comun * 100, 1), "%")


# ==============================================================
# PASO 5: VARIABLES MAS IMPORTANTES 
# ==============================================================
print("\n=== IMPORTANCIA DE CADA VARIABLE (en %) ===")
importancias = arbol.feature_importances_   # Lista con la importancia de cada columna
nombres = datos.columns                      # Lista con el nombre de cada columna

# Se arman pares (importancia, nombre) y se ordenan de mayor a menor
pares = sorted(zip(importancias, nombres), reverse=True)
for importancia, nombre in pares:
    if importancia > 0:   #  Muestra solo las que el arbol uso
        print(nombre, ":", round(importancia * 100, 1), "%")

# ==============================================================
# PASO 6: DATOS PARA LAS RECOMENDACIONES 
# Se calcula el % de abandono segun distintos grupos
# ==============================================================
tabla["abandono"] = resultado   # Agregamos la columna 1/0 a la tabla 

print("\n=== % DE ABANDONO POR GRUPO ===")
for columna in ["genero", "trabaja/NoTrabaja", "ActividadesExtracurriculares(Estudio)",
                "CantMateriasAprobadasPrimerCuatrimestre", "carrera"]:
    print((tabla.groupby(columna)["abandono"].mean() * 100).round(1).to_string())