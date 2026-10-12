# %% =====================================================
# PROYECTO 0 - PUMP FAILURE PREDICTION
# Predicción de fallos de bombas industriales
# ========================================================


# %% =====================================================
# FASE 1. IMPORTACIÓN DE LIBRERÍAS Y DATOS
# ========================================================

from pathlib import Path
import pandas as pd

# Definimos la carpeta del proyecto
project_path = Path(
    "C:/Users/david/Documents/PROYECTOS/pump_failure_project"
)

# Construimos la ruta del fichero Excel
data_path = project_path / "pump_failure_project.xlsx"

print("Ruta del archivo:", data_path)

# Comprobamos que el archivo existe
if not data_path.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo: {data_path}"
    )

# Leemos los datos originales
df_raw = pd.read_excel(data_path)

# Creamos una copia para trabajar
df = df_raw.copy()


# %% =====================================================
# FASE 2. EXPLORACIÓN INICIAL - EDA
# ========================================================

# Primeras filas
print("\nPRIMERAS FILAS:")
print(df.head())

# Dimensiones
print("\nDIMENSIONES:")
print(df.shape)

# Columnas
print("\nCOLUMNAS:")
print(df.columns)

# Información general
print("\nINFORMACIÓN DEL DATASET:")
df.info()

# Valores ausentes
print("\nVALORES AUSENTES:")
print(df.isna().sum())


# %% 2.1. ANÁLISIS DE VARIABLES CATEGÓRICAS

print("\nTIPOS DE BOMBA:")
print(df["pump_type"].value_counts())

print("\nPLANTAS:")
print(df["plant"].value_counts())

print("\nIDENTIFICADORES DE BOMBA:")
print(df["pump_id"].unique())

print("\nNÚMERO DE BOMBAS DIFERENTES:")
print(df["pump_id"].nunique())


# %% 2.2. VARIABLE OBJETIVO

print("\nDISTRIBUCIÓN DE FALLOS:")
print(
    df["failure_next_30_days"].value_counts()
)

print("\nDISTRIBUCIÓN PORCENTUAL:")
print(
    df["failure_next_30_days"]
    .value_counts(normalize=True)
    * 100
)


# %% 2.3. ESTADÍSTICA DESCRIPTIVA

print("\nESTADÍSTICA DESCRIPTIVA:")
print(df.describe())


# %% =====================================================
# FASE 3. LIMPIEZA DE DATOS
# ========================================================

# Convertimos timestamp a tipo datetime
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print("\nFECHAS INVÁLIDAS:")
print(df["timestamp"].isna().sum())

print("\nTIPO DE TIMESTAMP:")
print(df["timestamp"].dtype)


# %% 3.1. NORMALIZACIÓN DE VARIABLES CATEGÓRICAS

# Eliminamos espacios y convertimos a minúsculas
df["plant"] = (
    df["plant"]
    .str.strip()
    .str.lower()
)

df["pump_type"] = (
    df["pump_type"]
    .str.strip()
    .str.lower()
)

print("\nPLANTAS LIMPIAS:")
print(df["plant"].value_counts())

print("\nTIPOS DE BOMBA LIMPIOS:")
print(df["pump_type"].value_counts())


# %% 3.2. ORDENACIÓN CRONOLÓGICA

# Copia de los datos ya limpiados
df_processed = df.copy()

# Eliminamos registros con fecha inválida
df_processed = df_processed.dropna(
    subset=["timestamp"]
)

# Ordenamos cronológicamente
df_processed = (
    df_processed
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print("\nPRIMERAS FECHAS:")
print(
    df_processed[
        ["timestamp", "pump_id"]
    ].head()
)

print("\nÚLTIMAS FECHAS:")
print(
    df_processed[
        ["timestamp", "pump_id"]
    ].tail()
)


# %% =====================================================
# FASE 4. PREPARACIÓN PARA MACHINE LEARNING
# ========================================================

# Columnas identificativas
id_columns = [
    "pump_id",
    "timestamp"
]

# Variables numéricas
numeric_features = [
    "age_years",
    "operating_hours",
    "inlet_pressure_bar",
    "outlet_pressure_bar",
    "temperature_c",
    "vibration_mm_s",
    "flow_rate_m3_h",
    "days_since_maintenance",
    "failures_last_year"
]

# Variables categóricas
categorical_features = [
    "plant",
    "pump_type"
]

# Variables binarias
binary_features = [
    "alarm_last_7d"
]

# Variable objetivo
target = "failure_next_30_days"

# Todas las variables predictoras
feature_columns = (
    numeric_features
    + categorical_features
    + binary_features
)

print("\nVARIABLES PREDICTORAS:")
print(feature_columns)

print("\nNÚMERO DE VARIABLES:")
print(len(feature_columns))


# %% 4.1. SEPARACIÓN TEMPORAL TRAIN / TEST

# Percentil 80 de las fechas
cutoff_date = (
    df_processed["timestamp"]
    .quantile(0.80)
)

# Gap de seguridad
gap = pd.Timedelta(days=30)

# TRAIN
df_train = df_processed[
    df_processed["timestamp"]
    <= (cutoff_date - gap)
].copy()

# TEST
df_test = df_processed[
    df_processed["timestamp"]
    > cutoff_date
].copy()

# Reiniciamos índices
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("\nFECHA DE CORTE:")
print(cutoff_date)

print("\nGAP:")
print(gap)

print("\nTRAIN:")
print(df_train.shape)

print("\nTEST:")
print(df_test.shape)

print("\nÚLTIMA FECHA TRAIN:")
print(df_train["timestamp"].max())

print("\nPRIMERA FECHA TEST:")
print(df_test["timestamp"].min())


# Comprobaciones
assert not df_train.empty, (
    "TRAIN está vacío"
)

assert not df_test.empty, (
    "TEST está vacío"
)

assert (
    df_train["timestamp"].max() + gap
    <= df_test["timestamp"].min()
), "No se respeta la separación temporal"

print("\nSeparación temporal correcta.")


# %% 4.2. SEPARACIÓN FEATURES / TARGET

X_train = (
    df_train[feature_columns]
    .copy()
)

y_train = (
    df_train[target]
    .copy()
)

X_test = (
    df_test[feature_columns]
    .copy()
)

y_test = (
    df_test[target]
    .copy()
)

print("\nDIMENSIONES:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)

print("\nDISTRIBUCIÓN TRAIN:")
print(
    y_train.value_counts()
)

print("\nDISTRIBUCIÓN TEST:")
print(
    y_test.value_counts()
)

# Comprobamos que existen las dos clases
assert y_train.nunique() == 2, (
    "TRAIN no contiene ambas clases"
)

assert y_test.nunique() == 2, (
    "TEST no contiene ambas clases"
)


# %% 4.3. IMPORTACIÓN PREPROCESAMIENTO

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.pipeline import Pipeline

from sklearn.compose import (
    ColumnTransformer
)


# %% 4.4. PIPELINE NUMÉRICO

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# %% 4.5. PIPELINE CATEGÓRICO

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# %% 4.6. COLUMN TRANSFORMER

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        ),
        (
            "bin",
            "passthrough",
            binary_features
        )
    ]
)


# %% 4.7. APLICACIÓN DEL PREPROCESAMIENTO

# TRAIN aprende las transformaciones
X_train_ready = (
    preprocessor
    .fit_transform(X_train)
)

# TEST solo aplica las transformaciones
X_test_ready = (
    preprocessor
    .transform(X_test)
)

print("\nDATOS PREPROCESADOS:")

print(
    "X_train_ready:",
    X_train_ready.shape
)

print(
    "X_test_ready:",
    X_test_ready.shape
)


# %% =====================================================
# FASE 4.8. PRIMER MODELO
# ========================================================

from sklearn.linear_model import (
    LogisticRegression
)

model = LogisticRegression(
    max_iter=1000,
    random_state=45
)

# Entrenamiento
model.fit(
    X_train_ready,
    y_train
)

print(
    "\nEntrenamiento completado."
)


# %% 4.9. PREDICCIONES

y_pred = model.predict(
    X_test_ready
)

print(
    "\nPRIMERAS PREDICCIONES:"
)

print(
    y_pred[:10]
)


# %% 4.10. PROBABILIDADES

y_proba = model.predict_proba(
    X_test_ready
)

# Probabilidad de clase 1
failure_probability = (
    y_proba[:, 1]
)

print(
    "\nPRIMERAS PROBABILIDADES DE FALLO:"
)

print(
    failure_probability[:10]
)


# %% 4.11. TABLA DE RESULTADOS

df_results = (
    df_test[id_columns]
    .copy()
)

df_results["real"] = (
    y_test.to_numpy()
)

df_results["predicted"] = (
    y_pred
)

df_results[
    "failure_probability"
] = failure_probability

print(
    "\nRESULTADOS DEL MODELO:"
)

print(
    df_results.head(10)
)

print(
    "\nDIMENSIONES TABLA FINAL:"
)

print(
    df_results.shape
)


# %% =====================================================
# FASE 5. EVALUACIÓN DEL MODELO
# ========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


# %% 5.1. MATRIZ DE CONFUSIÓN

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(
    "\nMATRIZ DE CONFUSIÓN:"
)

print(cm)

tn, fp, fn, tp = (
    cm.ravel()
)

print(
    "\nVERDADEROS NEGATIVOS:",
    tn
)

print(
    "FALSOS POSITIVOS:",
    fp
)

print(
    "FALSOS NEGATIVOS:",
    fn
)

print(
    "VERDADEROS POSITIVOS:",
    tp
)


# %% 5.2. MÉTRICAS

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "\nEVALUACIÓN DEL MODELO"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1-score: {f1:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)


# %% 5.3. CLASSIFICATION REPORT

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Failure",
        "Failure"
    ],
    zero_division=0
)

print(
    "\nCLASSIFICATION REPORT:"
)

print(report)


# %% 5.4. MÉTRICAS BASADAS EN PROBABILIDADES

roc_auc = roc_auc_score(
    y_test,
    failure_probability
)

average_precision = (
    average_precision_score(
        y_test,
        failure_probability
    )
)

failure_rate = (
    y_test.mean()
)

print(
    "\nMÉTRICAS BASADAS EN PROBABILIDADES"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision:.4f}"
)

print(
    f"Porcentaje de fallos TEST: "
    f"{failure_rate:.4%}"
)


# %% 5.5. BASELINE DUMMY CLASSIFIER

from sklearn.dummy import (
    DummyClassifier
)

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(
    X_train_ready,
    y_train
)

dummy_pred = dummy_model.predict(
    X_test_ready
)

print(
    "\nCOMPARACIÓN DE MODELOS"
)

print(
    "Accuracy Logistic Regression:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "Accuracy Dummy:",
    accuracy_score(
        y_test,
        dummy_pred
    )
)

print(
    "Recall Logistic Regression:",
    recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "Recall Dummy:",
    recall_score(
        y_test,
        dummy_pred,
        zero_division=0
    )
)


# %% =====================================================
# FASE 6. MEJORA Y COMPARACIÓN DE MODELOS
# ========================================================


# %% 6.1. TRAIN / VALIDATION

# Percentil 80 del antiguo TRAIN
val_cutoff = (
    df_train["timestamp"]
    .quantile(0.80)
)

# Nuevo gap temporal
gap_val = pd.Timedelta(
    days=30
)

# Nuevo TRAIN
df_train_6 = df_train[
    df_train["timestamp"]
    <= (val_cutoff - gap_val)
].copy()

# VALIDATION
df_val_6 = df_train[
    df_train["timestamp"]
    > val_cutoff
].copy()

# Reiniciamos índices
df_train_6 = (
    df_train_6
    .reset_index(drop=True)
)

df_val_6 = (
    df_val_6
    .reset_index(drop=True)
)

print(
    "\nTRAIN:",
    df_train_6.shape
)

print(
    "VALIDATION:",
    df_val_6.shape
)

print(
    "TEST:",
    df_test.shape
)

print(
    "\nÚltima fecha TRAIN:",
    df_train_6[
        "timestamp"
    ].max()
)

print(
    "Primera fecha VAL:",
    df_val_6[
        "timestamp"
    ].min()
)

assert (
    df_train_6["timestamp"].max()
    + gap_val
    <= df_val_6["timestamp"].min()
), "No se respeta el gap temporal"


# %% 6.2. FEATURES Y TARGET

X_train_6 = (
    df_train_6[
        feature_columns
    ].copy()
)

y_train_6 = (
    df_train_6[
        target
    ].copy()
)

X_val_6 = (
    df_val_6[
        feature_columns
    ].copy()
)

y_val_6 = (
    df_val_6[
        target
    ].copy()
)

print(
    "\nX_train_6:",
    X_train_6.shape
)

print(
    "y_train_6:",
    y_train_6.shape
)

print(
    "X_val_6:",
    X_val_6.shape
)

print(
    "y_val_6:",
    y_val_6.shape
)

print(
    "\nDISTRIBUCIÓN TRAIN:"
)

print(
    y_train_6.value_counts()
)

print(
    "\nDISTRIBUCIÓN VALIDATION:"
)

print(
    y_val_6.value_counts()
)

assert (
    y_train_6.nunique() == 2
)

assert (
    y_val_6.nunique() == 2
)


# %% 6.3. CREACIÓN DE MODELOS

from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestClassifier
)

models = {

    # -----------------------------------------
    # MODELO 1
    # Regresión logística estándar
    # -----------------------------------------

    "LogReg_standard": Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=45
                )
            )
        ]
    ),

    # -----------------------------------------
    # MODELO 2
    # Regresión logística balanceada
    # -----------------------------------------

    "LogReg_balanced": Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=45
                )
            )
        ]
    ),

    # -----------------------------------------
    # MODELO 3
    # Random Forest balanceado
    # -----------------------------------------

    "RandomForest_balanced": Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight=(
                        "balanced_subsample"
                    ),
                    random_state=45,
                    n_jobs=-1
                )
            )
        ]
    )
}


# %% 6.4. ENTRENAMIENTO Y VALIDACIÓN

trained_models = {}

validation_probabilities = {}

for name, candidate in (
    models.items()
):

    print(
        "\nENTRENANDO:",
        name
    )

    # Entrenamiento
    candidate.fit(
        X_train_6,
        y_train_6
    )

    # Guardamos modelo entrenado
    trained_models[
        name
    ] = candidate

    # Probabilidades en VALIDATION
    val_proba = (
        candidate
        .predict_proba(
            X_val_6
        )[:, 1]
    )

    # Guardamos probabilidades
    validation_probabilities[
        name
    ] = val_proba

    # ROC-AUC
    roc = roc_auc_score(
        y_val_6,
        val_proba
    )

    # Average Precision
    ap = average_precision_score(
        y_val_6,
        val_proba
    )

    print(
        f"ROC-AUC: {roc:.4f}"
    )

    print(
        f"Average Precision: "
        f"{ap:.4f}"
    )


# %% =====================================================
# FASE 6.5. COMPARACIÓN DE THRESHOLDS
# ========================================================

from sklearn.metrics import (
    fbeta_score
)

# Umbrales que queremos estudiar
thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

# Aquí guardaremos cada experimento
rows = []

# Recorremos cada modelo
for name, val_proba in (
    validation_probabilities.items()
):

    # Métricas independientes
    # del threshold

    roc = roc_auc_score(
        y_val_6,
        val_proba
    )

    ap = average_precision_score(
        y_val_6,
        val_proba
    )

    # Ahora probamos cada threshold
    for threshold in thresholds:

        # Probabilidades -> 0 / 1
        y_pred_val = (
            val_proba
            >= threshold
        ).astype(int)

        # Matriz de confusión
        cm_val = confusion_matrix(
            y_val_6,
            y_pred_val,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm_val.ravel()
        )

        # Precision
        precision_val = (
            precision_score(
                y_val_6,
                y_pred_val,
                zero_division=0
            )
        )

        # Recall
        recall_val = (
            recall_score(
                y_val_6,
                y_pred_val,
                zero_division=0
            )
        )

        # F2
        f2_val = (
            fbeta_score(
                y_val_6,
                y_pred_val,
                beta=2,
                zero_division=0
            )
        )

        # Guardamos la configuración
        rows.append({
            "model": name,
            "threshold": threshold,

            "alarms": int(
                y_pred_val.sum()
            ),

            "precision": (
                precision_val
            ),

            "recall": (
                recall_val
            ),

            "f2": (
                f2_val
            ),

            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),

            "roc_auc": roc,

            "average_precision": ap
        })


# Convertimos la lista
# de diccionarios a DataFrame

df_comparison = pd.DataFrame(
    rows
)


# Ordenamos según F2

df_comparison = (
    df_comparison
    .sort_values(
        by=[
            "f2",
            "precision",
            "average_precision"
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nCOMPARACIÓN DE MODELOS "
    "Y THRESHOLDS"
)

print(
    df_comparison
    .round(3)
    .to_string(index=False)
)


# %% 6.5B. TOP 10 CONFIGURACIONES

print(
    "\nTOP 10 CONFIGURACIONES"
)

print(
    df_comparison[
        [
            "model",
            "threshold",
            "alarms",
            "precision",
            "recall",
            "f2",
            "tp",
            "fp",
            "fn"
        ]
    ]
    .head(10)
    .round(3)
    .to_string(index=False)
)


# %% =====================================================
# FASE 6.6. SELECCIÓN DEL MEJOR MODELO
# ========================================================

# Primera fila
# porque df_comparison ya está ordenado

best_row = (
    df_comparison.iloc[0]
)

# Nombre del modelo
best_model_name = (
    best_row["model"]
)

# Threshold ganador
best_threshold = float(
    best_row["threshold"]
)

# Recuperamos el modelo entrenado
best_model = (
    trained_models[
        best_model_name
    ]
)


print(
    "\nMEJOR CONFIGURACIÓN "
    "EN VALIDATION"
)

print(
    "Modelo:",
    best_model_name
)

print(
    "Threshold:",
    best_threshold
)

print(
    "Precision:",
    round(
        best_row["precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_row["recall"],
        4
    )
)

print(
    "F2-score:",
    round(
        best_row["f2"],
        4
    )
)

print(
    "Fallos detectados:",
    int(
        best_row["tp"]
    )
)

print(
    "Falsas alarmas:",
    int(
        best_row["fp"]
    )
)

print(
    "Fallos no detectados:",
    int(
        best_row["fn"]
    )
)


if best_row["f2"] == 0:

    print(
        "\nADVERTENCIA: ninguna "
        "configuración consiguió "
        "un F2 positivo."
    )


# %% =====================================================
# FASE 6.7. EVALUACIÓN FINAL SOBRE TEST
# ========================================================

# Datos TEST
X_test_6 = (
    df_test[
        feature_columns
    ].copy()
)

y_test_6 = (
    df_test[
        target
    ].copy()
)

# Probabilidad de fallo
test_proba = (
    best_model
    .predict_proba(
        X_test_6
    )[:, 1]
)

# Aplicamos el threshold
# seleccionado en VALIDATION
y_pred_test = (
    test_proba
    >= best_threshold
).astype(int)


# Matriz de confusión

cm_test = confusion_matrix(
    y_test_6,
    y_pred_test,
    labels=[0, 1]
)

tn, fp, fn, tp = (
    cm_test.ravel()
)


# Métricas

test_accuracy = (
    accuracy_score(
        y_test_6,
        y_pred_test
    )
)

test_precision = (
    precision_score(
        y_test_6,
        y_pred_test,
        zero_division=0
    )
)

test_recall = (
    recall_score(
        y_test_6,
        y_pred_test,
        zero_division=0
    )
)

test_f1 = (
    f1_score(
        y_test_6,
        y_pred_test,
        zero_division=0
    )
)

test_f2 = (
    fbeta_score(
        y_test_6,
        y_pred_test,
        beta=2,
        zero_division=0
    )
)

test_roc_auc = (
    roc_auc_score(
        y_test_6,
        test_proba
    )
)

test_ap = (
    average_precision_score(
        y_test_6,
        test_proba
    )
)


print(
    "\nRESULTADOS EN TEST"
)

print(
    "Modelo:",
    best_model_name
)

print(
    "Threshold:",
    best_threshold
)

print(
    "\nMATRIZ DE CONFUSIÓN:"
)

print(
    cm_test
)

print(
    "\nMÉTRICAS:"
)

print(
    f"Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{test_precision:.4f}"
)

print(
    f"Recall: "
    f"{test_recall:.4f}"
)

print(
    f"F1-score: "
    f"{test_f1:.4f}"
)

print(
    f"F2-score: "
    f"{test_f2:.4f}"
)

print(
    f"ROC-AUC: "
    f"{test_roc_auc:.4f}"
)

print(
    f"Average Precision: "
    f"{test_ap:.4f}"
)


print(
    "\nRESUMEN DE FALLOS:"
)

print(
    "Fallos detectados:",
    tp
)

print(
    "Fallos no detectados:",
    fn
)

print(
    "Falsas alarmas:",
    fp
)


print(
    "\nCLASSIFICATION REPORT:"
)

print(
    classification_report(
        y_test_6,
        y_pred_test,
        target_names=[
            "No Failure",
            "Failure"
        ],
        zero_division=0
    )
)


# %% =====================================================
# FASE 6.8. TABLA FINAL DE RESULTADOS
# ========================================================

df_results_6 = (
    df_test[
        id_columns
    ].copy()
)

# Resultado real
df_results_6[
    "real"
] = y_test_6.to_numpy()

# Predicción
df_results_6[
    "predicted"
] = y_pred_test

# Probabilidad
df_results_6[
    "failure_probability"
] = test_proba


# Por defecto consideramos
# todos TN

df_results_6[
    "result_type"
] = "TN"


# Falsos positivos
df_results_6.loc[
    (
        df_results_6["real"]
        == 0
    )
    &
    (
        df_results_6["predicted"]
        == 1
    ),
    "result_type"
] = "FP"


# Falsos negativos
df_results_6.loc[
    (
        df_results_6["real"]
        == 1
    )
    &
    (
        df_results_6["predicted"]
        == 0
    ),
    "result_type"
] = "FN"


# Verdaderos positivos
df_results_6.loc[
    (
        df_results_6["real"]
        == 1
    )
    &
    (
        df_results_6["predicted"]
        == 1
    ),
    "result_type"
] = "TP"


# Ordenamos por mayor
# probabilidad de fallo

df_results_6 = (
    df_results_6
    .sort_values(
        "failure_probability",
        ascending=False
    )
)


print(
    "\nOBSERVACIONES CON "
    "MAYOR RIESGO ESTIMADO:"
)

print(
    df_results_6.head(15)
)


print(
    "\nDISTRIBUCIÓN DE RESULTADOS:"
)

print(
    df_results_6[
        "result_type"
    ].value_counts()
)


# %% 6.9. ANALIZAR FALSOS NEGATIVOS

false_negatives = (
    df_results_6[
        df_results_6[
            "result_type"
        ] == "FN"
    ]
)

print(
    "\nFALSOS NEGATIVOS:"
)

print(
    false_negatives
)


# %% 6.10. ANALIZAR FALSOS POSITIVOS

false_positives = (
    df_results_6[
        df_results_6[
            "result_type"
        ] == "FP"
    ]
)

print(
    "\nFALSOS POSITIVOS:"
)

print(
    false_positives
)

# %% =====================================================
# FASE 7. IMPORTANCIA DE VARIABLES
# ========================================================

from sklearn.inspection import permutation_importance

importance_result = permutation_importance(
    best_model,
    X_test_6,
    y_test_6,
    scoring="average_precision",
    n_repeats=10,
    random_state=45,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance_result.importances_mean
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nIMPORTANCIA DE VARIABLES:")
print(feature_importance)

# %% =====================================================
# FASE 8. EXPORTACIÓN DE RESULTADOS
# ========================================================

# Creamos carpeta de resultados
output_path = project_path / "outputs"

output_path.mkdir(
    exist_ok=True
)

# Guardamos comparación de modelos
df_comparison.to_csv(
    output_path / "model_comparison.csv",
    index=False
)

# Guardamos predicciones finales
df_results_6.to_csv(
    output_path / "predictions.csv",
    index=False
)

# Guardamos importancia de variables
feature_importance.to_csv(
    output_path / "feature_importance.csv",
    index=False
)

# Guardamos fallos no detectados
false_negatives.to_csv(
    output_path / "false_negatives.csv",
    index=False
)

print("\nRESULTADOS GUARDADOS EN:")
print(output_path)

# %% =====================================================
# FASE 8B. VISUALIZACIÓN DE RESULTADOS
# ========================================================

import matplotlib.pyplot as plt

figures_path = output_path / "figures"

figures_path.mkdir(
    exist_ok=True
)


# %% 8B.1 - DISTRIBUCIÓN DE LA VARIABLE OBJETIVO

class_counts = (
    df_processed[target]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))

plt.bar(
    ["No Failure", "Failure"],
    class_counts.values
)

plt.title(
    "Distribución de la variable objetivo"
)

plt.ylabel(
    "Número de observaciones"
)

plt.xlabel(
    "Clase"
)

plt.tight_layout()

plt.savefig(
    figures_path / "target_distribution.png",
    dpi=300
)

plt.show()


# %% 8B.2 - MATRIZ DE CONFUSIÓN

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_test,
    display_labels=[
        "No Failure",
        "Failure"
    ]
)

disp.plot(
    values_format="d"
)

plt.title(
    "Matriz de confusión - Modelo final"
)

plt.tight_layout()

plt.savefig(
    figures_path / "confusion_matrix.png",
    dpi=300
)

plt.show()
# %% 8B.3 - EFECTO DEL THRESHOLD

threshold_results = (
    df_comparison[
        df_comparison["model"]
        == best_model_name
    ]
    .sort_values("threshold")
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["f2"],
    marker="o",
    label="F2"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Threshold elegido = {best_threshold}"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Métrica"
)

plt.title(
    "Efecto del threshold sobre el modelo"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    figures_path / "threshold_analysis.png",
    dpi=300
)

plt.show()

# %% 8B.4 - IMPORTANCIA DE VARIABLES

top_features = (
    feature_importance
    .head(10)
    .sort_values(
        "importance"
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Variable"
)

plt.title(
    "Variables más importantes del modelo"
)

plt.tight_layout()

plt.savefig(
    figures_path / "feature_importance.png",
    dpi=300
)

plt.show()


# %% 8B.5 - DISTRIBUCIÓN DE PROBABILIDADES

no_failure_prob = df_results_6.loc[
    df_results_6["real"] == 0,
    "failure_probability"
]

failure_prob = df_results_6.loc[
    df_results_6["real"] == 1,
    "failure_probability"
]

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    no_failure_prob,
    bins=20,
    alpha=0.6,
    label="No Failure"
)

plt.hist(
    failure_prob,
    bins=20,
    alpha=0.6,
    label="Failure"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlabel(
    "Probabilidad estimada de fallo"
)

plt.ylabel(
    "Número de observaciones"
)

plt.title(
    "Distribución del riesgo predicho"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    figures_path / "risk_distribution.png",
    dpi=300
)

plt.show()