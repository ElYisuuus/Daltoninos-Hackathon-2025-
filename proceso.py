import pandas as pd
import re
import os


# Configurar opciones de visualización de Pandas para mejor legibilidad
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.expand_frame_repr', False)


# --- CONFIGURACIÓN DE ARCHIVOS ---
INPUT_FILE = 'diabetes_training_data.csv'
OUTPUT_FILE = 'diabetes_features_final.csv'

# --- FUNCIONES AUXILIARES ---

def clean_text(text):
    """Limpia y normaliza el texto de las notas médicas."""
    text = str(text).lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-záéíóúüñ0-9\s.,]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numerical_value(note, pattern_group):
    """Extrae un valor numérico (entero o decimal) asociado a un patrón."""
    match = re.search(pattern_group, note)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def extract_binary_feature(note, positive_keywords):

    positive_pattern = r'\b(?:' + '|'.join(positive_keywords) + r')\b'
    
    negation_patterns = r'(?:no\s+|non\s*|not\s+|denies\s+|without\s+|absence\s+of\s+|no\s+history\s+of\s+|no\s+evidence\s+of\s*|\bno\b|\bnegativo\b|\bniega\b)'

    if re.search(positive_pattern, note):

        if re.search(negation_patterns + r'.{0,25}?' + positive_pattern, note):
            return 0
        else:
            return 1
    return 0

def extract_gender(note):
    note = note.lower()
    # Buscar patrones específicos como "X year old female/male"
    if re.search(r'(female|woman|mujer|femenino)', note):
        return 2
    if re.search(r'\bmale\b(?!.*female)', note):  # "male" pero NO si hay "female"
        return 1
    return 0

# --- PROCESO PRINCIPAL ---

def process_diabetes_data():
    """Ejecuta el pipeline completo de procesamiento de datos."""
    
    print("=" * 60)
    print("INICIANDO PIPELINE DE PROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    try:
        # =====================================================================
        # PASO 1: CARGAR DATOS Y REEMPLAZAR COMAS POR GUIONES
        # =====================================================================
        print(f"\n[1/5] Cargando archivo: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        print(f"✓ Archivo cargado exitosamente. Filas: {len(df)}")
        
        # Reemplazar comas por guiones en las notas médicas
        df['medical_note'] = df['medical_note'].str.replace(',', '-')
        print("✓ Comas convertidas a guiones en medical_note")
        
        # =====================================================================
        # PASO 2: RENOMBRAR Y LIMPIAR TEXTO
        # =====================================================================
        print("\n[2/5] Limpiando y normalizando texto...")
        df.rename(columns={'has_diabetes': 'target'}, inplace=True)
        df['cleaned_note'] = df['medical_note'].apply(clean_text)
        df['target'] = df['target'].astype(int)
        print("✓ Texto limpio y normalizado")
        
        # =====================================================================
        # PASO 3: EXTRACCIÓN DE FEATURES NUMÉRICAS
        # =====================================================================
        print("\n[3/5] Extrayendo features numéricas...")
        
        # BMI/IMC
        df['bmi_value'] = df['cleaned_note'].apply(
            lambda x: extract_numerical_value(x, r'(?:bmi|imc).*?(\d+\.?\d*)')
        )
        
        # HbA1c
        df['hba1c_value'] = df['cleaned_note'].apply(
            lambda x: extract_numerical_value(x, r'hba1c.*?(\d+\.?\d*)')
        )
        
        # Glucosa
        df['glucose_value'] = df['cleaned_note'].apply(
            lambda x: extract_numerical_value(x, r'(?:glucose|glucosa).*?(\d+\.?\d*)')
        )
        
        # Edad
        df['age_value'] = df['cleaned_note'].apply(
            lambda x: extract_numerical_value(x, r'\b(\d{1,3})\s*years?\s*old\b')
        )

        print("✓ Features numéricas extraídas (BMI, HbA1c, Glucosa, Edad)")
        print(f"  - BMI encontrados: {df['bmi_value'].notna().sum()}")
        print(f"  - HbA1c encontrados: {df['hba1c_value'].notna().sum()}")
        print(f"  - Glucosa encontrada: {df['glucose_value'].notna().sum()}")
        print(f"  - Edad encontrada: {df['age_value'].notna().sum()}")
        
        # =====================================================================
        # PASO 4: EXTRACCIÓN DE FEATURES BINARIAS
        # =====================================================================
        print("\n[4/5] Extrayendo features binarias...")
        
        # Fumar/Tabaquismo
        smoking_keywords = ['smoker', 'fumeur', 'raucher', 'fumador', 'smoking', 'tabaquismo']
        df['feature_smoker'] = df['cleaned_note'].apply(
            lambda x: extract_binary_feature(x, smoking_keywords)
        )

        # Hipertensión
        hypertension_keywords = ['hypertension', 'hipertensión', 'hypertonie', 'ipertensione', 'hta']
        df['feature_hypertension'] = df['cleaned_note'].apply(
            lambda x: extract_binary_feature(x, hypertension_keywords)
        )

        # Enfermedad cardíaca
        heart_keywords = ['heart disease', 'coronary', 'cardiac', 'infarto', 'angina', 'myocardial']
        df['feature_heart_disease'] = df['cleaned_note'].apply(
            lambda x: extract_binary_feature(x, heart_keywords)
        )
        
        df['feature_gender'] = df['cleaned_note'].apply(extract_gender)

        print("✓ Features binarias extraídas (Fumador, Hipertensión, Enfermedad cardíaca, Género)")
        print(f"  - Fumadores: {(df['feature_smoker'] == 1).sum()}")
        print(f"  - Con hipertensión: {(df['feature_hypertension'] == 1).sum()}")
        print(f"  - Con enfermedad cardíaca: {(df['feature_heart_disease'] == 1).sum()}")
        print(f"  - Género detectado: {(df['feature_gender'] > 0).sum()}")
        
        # =====================================================================
        # PASO 5: RELLENAR VALORES NULOS
        # =====================================================================
        print("\n[5/5] Rellenando valores nulos...")
        
        # Columnas finales a conservar
        columns_to_save = ['patient_id', 'target', 'cleaned_note',
                           'bmi_value', 'hba1c_value', 'glucose_value',
                           'age_value',
                           'feature_smoker', 'feature_hypertension',
                           'feature_heart_disease', 'feature_gender']
        
        df_final = df[columns_to_save].copy()
        
        # Rellenar nulos en columnas de texto
        text_cols = df_final.select_dtypes(include=['object']).columns
        df_final[text_cols] = df_final[text_cols].fillna('missing')
        
        # Rellenar nulos en columnas numéricas
        numeric_cols = df_final.select_dtypes(include=['number']).columns
        df_final[numeric_cols] = df_final[numeric_cols].fillna(0)
        
        # Asegurar que las columnas binarias sean enteros limpios
        df_final['feature_smoker'] = df_final['feature_smoker'].astype(int)
        df_final['feature_hypertension'] = df_final['feature_hypertension'].astype(int)
        df_final['feature_heart_disease'] = df_final['feature_heart_disease'].astype(int)
        df_final['feature_gender'] = df_final['feature_gender'].astype(int)

        print(f"✓ Valores nulos rellenados")
        
        # =====================================================================
        # GUARDAR ARCHIVO FINAL
        # =====================================================================
        # Ordenar por target y patient_id
        df_final = df_final.sort_values(by=['target', 'patient_id'], ascending=[True, True])
        
        output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
        df_final.to_csv(output_path, index=False, encoding='utf-8')
        
        # =====================================================================
        # RESUMEN FINAL
        # =====================================================================
        print("\n" + "=" * 60)
        print("✓ PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"Archivo guardado: {OUTPUT_FILE}")
        print(f"Ruta completa: {output_path}")
        print(f"\nTotal de registros: {len(df_final)}")
        print(f"Distribución de target:")
        print(f"  - Sin diabetes (0): {(df_final['target'] == 0).sum()}")
        print(f"  - Con diabetes (1): {(df_final['target'] == 1).sum()}")
        print(f"\nValores nulos restantes: {df_final.isnull().sum().sum()}")
        print("\n" + "=" * 60)
        print("Vista previa del DataFrame final:")
        print("=" * 60)
        print(df_final.head(10))
        print("\n" + "=" * 60)
        print("Información del DataFrame:")
        print("=" * 60)
        df_final.info()
        
        # Verificación final de valores nulos por columna
        print("\n" + "=" * 60)
        print("Conteo de valores nulos por columna:")
        print("=" * 60)
        print(df_final.isnull().sum())

    except FileNotFoundError:
        print(f"\nERROR: Archivo '{INPUT_FILE}' no encontrado.")
        print("Asegúrate de que el archivo esté en el directorio C:\\test")
    except PermissionError:
        print("\nERROR: No hay permisos para guardar el archivo.")
        print("Intenta ejecutar el script desde otra ubicación.")
    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

# --- EJECUTAR EL PIPELINE ---
if __name__ == "__main__":
    process_diabetes_data()