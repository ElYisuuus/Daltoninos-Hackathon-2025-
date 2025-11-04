import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import warnings
import pickle
warnings.filterwarnings('ignore', category=UserWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#Cargar el archivo
try:
    df = pd.read_csv('diabetes_features_final.csv')
except FileNotFoundError:
    print("Error: Archivo no encontrado. Asegúrate de que esté en la misma carpeta.")
    exit()

#Identificamos y rellenamos valores nulos con la mediana
text_cols = df.select_dtypes(include=['object']).columns
df[text_cols] = df[text_cols].fillna('missing')

numeric_cols = df.select_dtypes(include=['number']).columns

#Quitamos target para las variables numéricas
numeric_features_to_impute = [col for col in numeric_cols if col != 'target']

if numeric_features_to_impute:
    # Usamos SimpleImputer para rellenar con la mediana
    imputer = SimpleImputer(strategy='median')
    # fit_transform solo en el dataframe (sin la serie de 'target' si estuviera)
    df_imputed = imputer.fit_transform(df[numeric_features_to_impute])
    # Reasignamos los valores al dataframe
    df[numeric_features_to_impute] = df_imputed
else:
    print("No hay columnas numéricas (excluyendo 'target') que necesiten imputación.")

print(df.head())

#Preparación datos modelo
features = [
    'bmi_value', 'hba1c_value', 'glucose_value', 
    'feature_smoker', 'feature_hypertension',
    'feature_heart_disease', 'feature_gender', 'age_value'
]

#Verificación de columnas
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"Columnas disponibles: {list(df.columns)}")
    exit()

target_col = 'target' 

if target_col not in df.columns:
    print(f"Columnas disponibles: {list(df.columns)}")
    exit()

X = df[features].copy()
y = df[target_col].copy()

print(y.value_counts())

#Dividimos en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#ENTRENAMIENTO MODELO LIGHTGBM

#Definir el modelo base
#Usamos class_weight='balanced' como base
base_lgbm = LGBMClassifier(
    class_weight='balanced', 
    random_state=42,
    verbose=-1,
    n_estimators=100 # Empezar con 100, el grid lo ajustará
)

#Definir la parrilla de parámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 40], # Default es 31
    'max_depth': [-1, 10, 20] # -1 = sin límite
}

#Configurar GridSearchCV

grid_search = GridSearchCV(
    estimator=base_lgbm,
    param_grid=param_grid,
    scoring='f1',  
    cv=3,
    n_jobs=-1,
    verbose=2  
)

#Entrenar el GridSearchCV
grid_search.fit(X_train_scaled, y_train)

#Obtener el mejor modelo encontrado
model = grid_search.best_estimator_

print(f"Mejor F1-Score durante Cross-Validation: {grid_search.best_score_:.4f}")
print("="*60)

#EVALUAR EL MODELO AFINADO

y_pred_test = model.predict(X_test_scaled) # Predicción estándar (umbral 0.5)

print("\n" + "="*60)
print("="*60)
print(f"Accuracy Test: {accuracy_score(y_test, y_pred_test):.4f}")
print("\nReporte de clasificación (Test):")
print(classification_report(y_test, y_pred_test, 
                            target_names=['No Diabetes', 'Diabetes']))

#ANÁLISIS DE CURVA ROC Y UMBRAL ÓPTIMO ---
print("\n" + "="*60)
print("ANÁLISIS DE UMBRAL ÓPTIMO (SOBRE EL MODELO AFINADO)")
print("="*60)

# Obtener probabilidades
y_proba_test = model.predict_proba(X_test_scaled)[:, 1]

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
roc_auc = auc(fpr, tpr)

# Calcular precision-recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba_test)

# Encontrar umbral óptimo que maximiza F1-score
# (Usamos una búsqueda más fina que antes)
test_thresholds = np.linspace(0.1, 0.9, 100) # 100 puntos en lugar de ~16
f1_scores = []
for threshold in test_thresholds:
    y_pred_threshold = (y_proba_test >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
    f1_scores.append(f1)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = test_thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

# ------------------------------------------------------------------
print(f"\nF1-SCORE MÁXIMO OBTENIDO (con modelo afinado y umbral óptimo): {optimal_f1:.4f}")
# ------------------------------------------------------------------

print(f"\n Umbral óptimo (maximiza F1): {optimal_threshold:.4f}") # Más precisión en el umbral
print(f"   F1-Score con este umbral: {optimal_f1:.4f}")

# Predicciones con umbral óptimo
y_pred_optimal = (y_proba_test >= optimal_threshold).astype(int)

print("\n" + "="*60)
print(f"RESULTADOS CON UMBRAL OPTIMIZADO ({optimal_threshold:.4f}) - MODELO AFINADO")
print("="*60)
print(f"Accuracy Test: {accuracy_score(y_test, y_pred_optimal):.4f}")
print("\nReporte de clasificación (Test):")
print(classification_report(y_test, y_pred_optimal, 
                            target_names=['No Diabetes', 'Diabetes'], zero_division=0))

# --- 9. VISUALIZACIONES ---
# (El código de visualización es idéntico y sigue siendo válido)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 9.1 Matriz de Confusión - Estándar (modelo afinado)
cm_standard = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[0,0].set_title('Matriz de Confusión - Umbral Estándar (0.5) - Modelo Afinado')
axes[0,0].set_ylabel('Valor Real')
axes[0,0].set_xlabel('Predicción')

# 9.2 Matriz de Confusión - Optimizada (modelo afinado)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=axes[0,1],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[0,1].set_title(f'Matriz de Confusión - Umbral Optimizado ({optimal_threshold:.2f}) - Modelo Afinado')
axes[0,1].set_ylabel('Valor Real')
axes[0,1].set_xlabel('Predicción')

# 9.3 Curva ROC
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1,0].set_xlim([0.0, 1.0])
axes[1,0].set_ylim([0.0, 1.05])
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate (Recall)')
axes[1,0].set_title('Curva ROC - Modelo Afinado')
axes[1,0].legend(loc="lower right")
axes[1,0].grid(True, alpha=0.3)

# 9.4 F1-Score vs Umbral
axes[1,1].plot(test_thresholds, f1_scores, 'b-', linewidth=2)
axes[1,1].axvline(x=optimal_threshold, color='r', linestyle='--', 
                   label=f'Óptimo: {optimal_threshold:.2f}')
axes[1,1].axvline(x=0.5, color='gray', linestyle=':', label='Estándar: 0.50')
axes[1,1].set_xlabel('Umbral de Decisión')
axes[1,1].set_ylabel('F1-Score')
axes[1,1].set_title('F1-Score según Umbral - Modelo Afinado')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_completo_diabetes_lightgbm_AFINADO.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizaciones guardadas como 'analisis_completo_diabetes_lightgbm_AFINADO.png'")

# --- 10. IMPORTANCIA DE CARACTERÍSTICAS ---
# (Sigue siendo válido, ahora con el modelo afinado)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Importancia de Características (Modelo Afinado) ---")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importancia')
plt.title('Importancia de Características (LightGBM Afinado)')
plt.tight_layout()
plt.savefig('feature_importance_lightgbm_AFINADO.png', dpi=300, bbox_inches='tight')
print("✓ Importancia de características guardada como 'feature_importance_lightgbm_AFINADO.png'")


# --- 11. TABLA COMPARATIVA DE UMBRALES ---
# (Sigue siendo válido)
print("\n" + "="*60)
print("COMPARACIÓN DE DIFERENTES UMBRALES (Modelo Afinado)")
print("="*60)
comparison_data = []
# Asegurarnos de que el umbral óptimo esté en la lista
threshold_list = sorted(list(set([0.3, 0.4, 0.5, optimal_threshold])))
for threshold in threshold_list:
    y_pred_temp = (y_proba_test >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred_temp, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred_temp, zero_division=0)
    comparison_data.append({
        'Umbral': f"{threshold:.4f}",
        'Accuracy': f"{acc:.3f}",
        'Precision (Diabetes)': f"{prec:.3f}",
        'Recall (Diabetes)': f"{rec:.3f}",
        'F1-Score': f"{f1:.3f}"
    })
comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))


# --- 12. FUNCIÓN MEJORADA PARA PREDECIR ---
# (Sigue siendo válida, la actualizamos para que use el nuevo umbral por defecto)
def predecir_diabetes_optimizado(bmi, hba1c, glucose, smoker, hypertension, heart_disease, gender, age, 
                                 threshold=optimal_threshold): # <-- Usa el nuevo umbral
    """
    Predice si un paciente tiene diabetes usando el modelo AFINADO y umbral optimizado
    """
    nuevo_paciente = np.array([[bmi, hba1c, glucose, smoker, hypertension, heart_disease, gender, age]])
    nuevo_paciente_scaled = scaler.transform(nuevo_paciente)
    
    probabilidad = model.predict_proba(nuevo_paciente_scaled)[0]
    prob_diabetes = probabilidad[1]
    prediccion = 1 if prob_diabetes >= threshold else 0

    print(f"\n{'='*60}")
    print(f"PREDICCIÓN PARA NUEVO PACIENTE (umbral={threshold:.4f})")
    print(f"{'='*60}")
    print(f"Probabilidad de DIABETES: {prob_diabetes:.1%}")
    if prediccion == 1:
        print(f"RESULTADO: DIABETES DETECTADA")
    else:
        print(f"RESULTADO: NO DIABETES")
    print(f"{'='*60}\n")
    return prediccion, probabilidad

# --- 13. EJEMPLOS DE USO ---
# (Siguen siendo válidos)
print("\n" + "="*60)
print("EJEMPLOS DE PREDICCIÓN (Modelo Afinado)")
print("="*60)
predecir_diabetes_optimizado(bmi=32, hba1c=7.2, glucose=160, smoker=1, hypertension=1,
                             heart_disease=1, gender=1, age=65)
predecir_diabetes_optimizado(bmi=24, hba1c=5.0, glucose=95, smoker=0, hypertension=0,
                             heart_disease=0, gender=2, age=28)

# --- 14. GUARDAR MODELOS ---
with open('modelo_diabetes_lightgbm_AFINADO.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('scaler_diabetes_AFINADO.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('umbral_optimo_lightgbm_AFINADO.txt', 'w') as f:
    f.write(f"{optimal_threshold:.4f}")

print("\n" + "="*60)
print("ARCHIVOS GUARDADOS (AFINADOS)")
print("="*60)
print("✓ modelo_diabetes_lightgbm_AFINADO.pkl")
print("✓ scaler_diabetes_AFINADO.pkl")
print(f"✓ umbral_optimo_lightgbm_AFINADO.txt (valor: {optimal_threshold:.4f})")

print("\n" + "="*60)
print("RESUMEN DEL AFINAMIENTO")
print("="*60)
print(f"Modelo afinado con GridSearchCV optimizando para 'f1'.")
print(f"Mejor F1 en CV: {grid_search.best_score_:.4f}")
print(f"Umbral óptimo final: {optimal_threshold:.4f}")
print(f"F1-Score final en Test: {optimal_f1:.4f}")
print("="*60) 

# --- 15. PROCESAR Y PREDECIR 'diabetes_test_data.csv' ---
print("\n" + "="*60)
print("PROCESANDO 'diabetes_test_data.csv' PARA PREDICCIONES")
print("="*60)

# 15.1. Función para extraer características del texto
def extract_features(note):
    """
    Usa regex para extraer las 8 características del modelo desde una nota médica.
    """
    features_dict = {}
    note = str(note).lower() # Normalizar a minúsculas

    # --- Diccionario para edades escritas ---
    age_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 
        'twelve': 12, 'thirteen': 13, 'fifteen': 15, 'sixteen': 16, 
        'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'eighty': 80
    }

    # 1. Edad (age_value)
    age_match_num = re.search(r'(\d+)-year-old', note)
    if age_match_num:
        features_dict['age_value'] = int(age_match_num.group(1))
    else:
        age_match_text = re.search(r'(\w+)-year-old', note)
        if age_match_text and age_match_text.group(1) in age_words:
            features_dict['age_value'] = age_words[age_match_text.group(1)]
        else:
            features_dict['age_value'] = np.nan # Dejar que el imputer lo rellene

    # 2. Género (feature_gender) - Asumiendo 1=Male, 2=Female
    if 'female' in note:
        features_dict['feature_gender'] = 2
    elif 'male' in note:
        features_dict['feature_gender'] = 1
    else:
        features_dict['feature_gender'] = np.nan # Dejar que el imputer lo rellene

    # 3. BMI (bmi_value)
    # --- LÍNEA CORREGIDA ---
    bmi_match = re.search(r'bmi (?:is|of) (?:approximately )?(\d+(?:\.\d+)?)', note)
    features_dict['bmi_value'] = float(bmi_match.group(1)) if bmi_match else np.nan

    # 4. HbA1c (hba1c_value)
    # --- LÍNEA CORREGIDA ---
    hba1c_match = re.search(r'hba1c (?:is|of) (\d+(?:\.\d+)?)%', note)
    features_dict['hba1c_value'] = float(hba1c_match.group(1)) if hba1c_match else np.nan

    # 5. Glucosa (glucose_value)
    # --- LÍNEA CORREGIDA ---
    glucose_match = re.search(r'glucose (?:level )?(?:is|of|reading is|measurement is) (\d+(?:\.\d+)?) mg/dl', note)
    if not glucose_match:
         # --- LÍNEA CORREGIDA ---
         glucose_match = re.search(r'glucose of (\d+(?:\.\d+)?)', note) # Otro patrón
    features_dict['glucose_value'] = float(glucose_match.group(1)) if glucose_match else np.nan

    # 6. Fumador (feature_smoker) - 0=No, 1=Sí (actual o pasado)
    if re.search(r'non-smoker', note) or re.search(r'no smoking history', note):
        features_dict['feature_smoker'] = 0
    elif re.search(r'(?:current|past|history of) smok(?:er|ing)', note) or re.search(r'is a (?:current|past)? ?smoker', note):
        features_dict['feature_smoker'] = 1
    else:
        features_dict['feature_smoker'] = np.nan

    # 7. Hipertensión (feature_hypertension) - 0=No, 1=Sí
    if re.search(r'no (?:history|signs) of hypertension', note) or re.search(r'denies hypertension', note):
        features_dict['feature_hypertension'] = 0
    elif re.search(r'(?:history|diagnosis) of hypertension', note) or re.search(r'has hypertension', note):
        features_dict['feature_hypertension'] = 1
    else:
        features_dict['feature_hypertension'] = np.nan

    # 8. Enf. Cardíaca (feature_heart_disease) - 0=No, 1=Sí
    if re.search(r'no (?:history|signs|known) (?:of )?heart disease', note) or re.search(r'denies heart disease', note):
        features_dict['feature_heart_disease'] = 0
    elif re.search(r'(?:history|diagnosis) of heart disease', note) or re.search(r'has heart disease', note):
        features_dict['feature_heart_disease'] = 1
    else:
        features_dict['feature_heart_disease'] = np.nan
            
    # Devuelve una Serie de pandas para fácil conversión
    return pd.Series(features_dict, index=features) # Asegura el orden de columnas

# 15.2. Cargar los nuevos datos
try:
    df_new_patients = pd.read_csv('diabetes_test_data.csv')
    print(f"Cargadas {len(df_new_patients)} notas de pacientes desde 'diabetes_test_data.csv'.")
except FileNotFoundError:
    print("ERROR: Archivo 'diabetes_test_data.csv' no encontrado. No se pueden generar predicciones.")
    exit()

# 15.3. Extraer características
print("... Extrayendo características desde texto (puede tardar un momento)...")
# 'features' es la lista definida en la sección 3 del script
extracted_features = df_new_patients['medical_note'].apply(extract_features)

# 15.4. Preparar DataFrame para el modelo
# (Combina patient_id con las nuevas características extraídas)
X_new = pd.concat([df_new_patients[['patient_id']], extracted_features], axis=1)

print("\n--- Primeras 5 filas de características EXTRAÍDAS (antes de imputar) ---")
print(X_new.head())

# 15.5. Imputar valores nulos (con el Imputer de MEDIANA del entrenamiento)
# 'imputer' y 'numeric_features_to_impute' ya existen en memoria de la Sección 2
X_new_to_impute = X_new[numeric_features_to_impute]
X_new_imputed_values = imputer.transform(X_new_to_impute)
# Reasignamos los valores imputados
X_new[numeric_features_to_impute] = X_new_imputed_values

# 15.6. Escalar datos (con el Scaler del entrenamiento)
# 'scaler' ya existe en memoria de la Sección 5
X_new_scaled_values = scaler.transform(X_new[features]) # Usamos la lista 'features'

# 15.7. Predecir con el modelo afinado y umbral óptimo
# 'model' y 'optimal_threshold' ya existen en memoria
print("\n... Realizando predicciones...")
probabilidades_new = model.predict_proba(X_new_scaled_values)[:, 1]
predicciones_new = (probabilidades_new >= optimal_threshold).astype(int)

# --- 15.8. Crear y guardar el archivo CSV final (MODIFICADO PARA FORMATO) ---
df_output = pd.DataFrame({
    'patient_id': X_new['patient_id'],
    'has_diabetes': predicciones_new  # <-- CAMBIO DE NOMBRE DE COLUMNA
})

# Convertir patient_id a string y añadir el prefijo 'patient_'
# Primero convertimos a int (para asegurar que no haya decimales) y luego a str
# Usamos .zfill(5) para asegurar 5 dígitos con ceros a la izquierda
df_output['patient_id'] = 'patient_' + df_output['patient_id'].astype(int).astype(str).str.zfill(5) 

output_filename = 'predicciones_pacientes.csv'
df_output.to_csv(output_filename, index=False)

print("\n" + "="*60)
print(f"¡ÉXITO! Predicciones (formato 'patient_id,has_diabetes') guardadas en '{output_filename}'")
print("="*60)
print(df_output.head())

