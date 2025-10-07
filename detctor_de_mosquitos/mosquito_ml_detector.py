import cv2
import numpy as np

def get_window_flag():
    """Obtiene la bandera correcta para ventanas según la versión de OpenCV"""
    if hasattr(cv2, 'WINDOW_NORMAL'):
        return cv2.WINDOW_NORMAL
    elif hasattr(cv2, 'WINDOW_RESIZABLE'):
        return cv2.WINDOW_NORMAL
    elif hasattr(cv2, 'WINDOW_AUTOSIZE'):
        return cv2.WINDOW_AUTOSIZE
    else:
        return 1

def create_compatible_window(window_name):
    """Crea ventana compatible con diferentes versiones de OpenCV"""
    try:
        cv2.namedWindow(window_name, get_window_flag())
    except:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except:
            cv2.namedWindow(window_name)

import numpy as np
import cv2
import os
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import deque
import threading
import json

class DetectorMosquitoML:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.modelo_entrenado = False
        
        # Configuración de cámara
        self.cap = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Parámetros de detección
        self.MIN_AREA = 5
        self.MAX_AREA = 500
        
        # Historial de detecciones para tracking
        self.tracking_history = deque(maxlen=20)
        self.detection_confidence_threshold = 0.7
        
        # Estadísticas
        self.stats = {
            'total_detections': 0,
            'confirmed_mosquitos': 0,
            'false_positives': 0,
            'detection_times': []
        }
    
    def extraer_caracteristicas(self, contorno, frame_gray, movimiento_info=None):
        """
        Extrae características de un contorno para clasificación
        """
        caracteristicas = []
        
        # 1. Características geométricas básicas
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        
        # Evitar división por cero
        if perimetro == 0:
            return None
            
        # Relación área/perímetro
        area_perimetro_ratio = area / (perimetro ** 2) if perimetro > 0 else 0
        caracteristicas.extend([area, perimetro, area_perimetro_ratio])
        
        # 2. Características del rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        caracteristicas.extend([w, h, aspect_ratio, extent])
        
        # 3. Características de forma
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Compacidad
        compactness = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
        caracteristicas.extend([solidity, compactness])
        
        # 4. Momentos de Hu (invariantes a rotación, escala y traslación)
        momentos = cv2.moments(contorno)
        if momentos['m00'] != 0:
            hu_moments = cv2.HuMoments(momentos)
            # Convertir a log para estabilidad numérica
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            caracteristicas.extend(hu_moments.flatten())
        else:
            caracteristicas.extend([0] * 7)  # 7 momentos de Hu
        
        # 5. Características de textura (usando la región del contorno)
        mask = np.zeros(frame_gray.shape, np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)
        
        # Estadísticas de intensidad
        intensidades = frame_gray[mask == 255]
        if len(intensidades) > 0:
            mean_intensity = np.mean(intensidades)
            std_intensity = np.std(intensidades)
            caracteristicas.extend([mean_intensity, std_intensity])
        else:
            caracteristicas.extend([0, 0])
        
        # 6. Características de movimiento (si están disponibles)
        if movimiento_info:
            velocidad = movimiento_info.get('velocidad', 0)
            aceleracion = movimiento_info.get('aceleracion', 0)
            cambios_direccion = movimiento_info.get('cambios_direccion', 0)
            caracteristicas.extend([velocidad, aceleracion, cambios_direccion])
        else:
            caracteristicas.extend([0, 0, 0])
        
        # 7. Características adicionales específicas para mosquitos
        # Relación de dimensiones (los mosquitos tienden a ser alargados)
        elongacion = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        
        # Densidad de píxeles
        densidad = np.sum(mask == 255) / (w * h) if (w * h) > 0 else 0
        
        caracteristicas.extend([elongacion, densidad])
        
        return np.array(caracteristicas)
    
    def calcular_movimiento(self, centro_actual, timestamp):
        """
        Calcula características de movimiento basadas en posiciones anteriores
        """
        self.tracking_history.append({'centro': centro_actual, 'tiempo': timestamp})
        
        if len(self.tracking_history) < 3:
            return {'velocidad': 0, 'aceleracion': 0, 'cambios_direccion': 0}
        
        # Calcular velocidades
        velocidades = []
        for i in range(1, len(self.tracking_history)):
            p1 = self.tracking_history[i-1]
            p2 = self.tracking_history[i]
            
            distancia = np.sqrt((p2['centro'][0] - p1['centro'][0])**2 + 
                              (p2['centro'][1] - p1['centro'][1])**2)
            tiempo_diff = p2['tiempo'] - p1['tiempo']
            velocidad = distancia / tiempo_diff if tiempo_diff > 0 else 0
            velocidades.append(velocidad)
        
        velocidad_promedio = np.mean(velocidades) if velocidades else 0
        
        # Calcular aceleración
        aceleracion = 0
        if len(velocidades) >= 2:
            aceleracion = np.std(velocidades)  # Variabilidad como proxy de aceleración
        
        # Contar cambios de dirección
        cambios_direccion = 0
        if len(self.tracking_history) >= 3:
            for i in range(2, len(self.tracking_history)):
                p1 = self.tracking_history[i-2]['centro']
                p2 = self.tracking_history[i-1]['centro']
                p3 = self.tracking_history[i]['centro']
                
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Producto punto normalizado
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                magnitude1 = np.sqrt(v1[0]**2 + v1[1]**2)
                magnitude2 = np.sqrt(v2[0]**2 + v2[1]**2)
                
                if magnitude1 > 0 and magnitude2 > 0:
                    cos_angle = dot_product / (magnitude1 * magnitude2)
                    if cos_angle < 0:  # Cambio de dirección significativo
                        cambios_direccion += 1
        
        return {
            'velocidad': velocidad_promedio,
            'aceleracion': aceleracion,
            'cambios_direccion': cambios_direccion
        }
    
    def recolectar_datos_entrenamiento(self, duracion_minutos=10):
        """
        Recolecta datos para entrenar el modelo
        """
        if self.cap is None:
            raise Exception("Cámara no inicializada")
        
        print(f"🎯 Iniciando recolección de datos por {duracion_minutos} minutos...")
        print("Instrucciones:")
        print("- Presiona 'm' cuando veas un MOSQUITO")
        print("- Presiona 'n' cuando veas un NO-MOSQUITO (polvo, insecto diferente, etc.)")
        print("- Presiona 'q' para terminar antes")
        
        datos_entrenamiento = []
        etiquetas = []
        
        fin_tiempo = time.time() + (duracion_minutos * 60)
        frame_count = 0
        
        cv2.namedWindow('Recolección de Datos', cv2.WINDOW_NORMAL)
        
        while time.time() < fin_tiempo:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplicar sustracción de fondo
            fg_mask = self.background_subtractor.apply(frame_gray)
            
            # Limpiar máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contornos, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objetos_detectados = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if self.MIN_AREA < area < self.MAX_AREA:
                    objetos_detectados.append(contorno)
            
            # Dibujar contornos detectados
            frame_display = frame.copy()
            for i, contorno in enumerate(objetos_detectados):
                cv2.drawContours(frame_display, [contorno], -1, (0, 255, 0), 2)
                
                # Mostrar número del objeto
                M = cv2.moments(contorno)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame_display, str(i), (cx, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar información
            tiempo_restante = (fin_tiempo - time.time()) / 60
            info_text = f"Tiempo: {tiempo_restante:.1f}min | Muestras: {len(datos_entrenamiento)} | Objetos: {len(objetos_detectados)}"
            cv2.putText(frame_display, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Recolección de Datos', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m') or key == ord('n'):
                # Etiquetar objetos detectados
                if objetos_detectados:
                    print(f"\n📊 Frame {frame_count}: {len(objetos_detectados)} objetos detectados")
                    for i, contorno in enumerate(objetos_detectados):
                        caracteristicas = self.extraer_caracteristicas(contorno, frame_gray)
                        if caracteristicas is not None:
                            datos_entrenamiento.append(caracteristicas)
                            etiqueta = 1 if key == ord('m') else 0  # 1=mosquito, 0=no-mosquito
                            etiquetas.append(etiqueta)
                            
                            tipo = "MOSQUITO" if etiqueta == 1 else "NO-MOSQUITO"
                            print(f"  Objeto {i}: {tipo}")
        
        cv2.destroyAllWindows()
        
        # Guardar datos
        if datos_entrenamiento:
            datos = {
                'caracteristicas': np.array(datos_entrenamiento),
                'etiquetas': np.array(etiquetas),
                'timestamp': time.time()
            }
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"datos_mosquitos_{timestamp}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(datos, f)
            
            print(f"\n✅ Datos guardados en: {filename}")
            print(f"📊 Total de muestras: {len(datos_entrenamiento)}")
            print(f"🦟 Mosquitos: {sum(etiquetas)}")
            print(f"❌ No-mosquitos: {len(etiquetas) - sum(etiquetas)}")
            
            return datos
        else:
            print("❌ No se recolectaron datos")
            return None
    
    def entrenar_modelo(self, datos_archivo=None, datos_dict=None):
        """
        Entrena el modelo de clasificación
        """
        # Cargar datos
        if datos_dict is not None:
            X = datos_dict['caracteristicas']
            y = datos_dict['etiquetas']
        elif datos_archivo is not None:
            with open(datos_archivo, 'rb') as f:
                datos = pickle.load(f)
            X = datos['caracteristicas']
            y = datos['etiquetas']
        else:
            raise ValueError("Debe proporcionar datos_archivo o datos_dict")
        
        if len(X) == 0:
            raise ValueError("No hay datos para entrenar")
        
        print(f"🎯 Entrenando modelo con {len(X)} muestras...")
        print(f"📊 Distribución: {sum(y)} mosquitos, {len(y) - sum(y)} no-mosquitos")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Probar diferentes modelos
        modelos = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        mejor_modelo = None
        mejor_score = 0
        resultados = {}
        
        for nombre, modelo in modelos.items():
            print(f"\n🔧 Entrenando {nombre}...")
            
            # Validación cruzada
            if len(np.unique(y_train)) > 1:  # Solo si hay ambas clases
                scores = cross_val_score(modelo, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
                score_promedio = np.mean(scores)
            else:
                score_promedio = 0
            
            # Entrenar en todo el conjunto de entrenamiento
            modelo.fit(X_train_scaled, y_train)
            
            # Evaluar en conjunto de prueba
            if len(X_test) > 0:
                score_test = modelo.score(X_test_scaled, y_test)
                y_pred = modelo.predict(X_test_scaled)
                
                print(f"   Validación cruzada: {score_promedio:.3f}")
                print(f"   Precisión en test: {score_test:.3f}")
                
                resultados[nombre] = {
                    'modelo': modelo,
                    'cv_score': score_promedio,
                    'test_score': score_test,
                    'predictions': y_pred,
                    'y_test': y_test
                }
                
                if score_test > mejor_score:
                    mejor_score = score_test
                    mejor_modelo = modelo
        
        if mejor_modelo is None:
            # Si no hay conjunto de test, usar el mejor por validación cruzada
            mejor_cv = max(resultados.values(), key=lambda x: x['cv_score'])
            mejor_modelo = mejor_cv['modelo']
        
        self.modelo = mejor_modelo
        self.modelo_entrenado = True
        
        # Guardar modelo
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        modelo_filename = f"modelo_mosquito_{timestamp}.pkl"
        
        with open(modelo_filename, 'wb') as f:
            pickle.dump({
                'modelo': self.modelo,
                'scaler': self.scaler,
                'timestamp': time.time(),
                'num_caracteristicas': X.shape[1]
            }, f)
        
        print(f"\n✅ Modelo entrenado y guardado en: {modelo_filename}")
        
        # Mostrar reporte detallado del mejor modelo
        if len(X_test) > 0 and len(np.unique(y_test)) > 1:
            nombre_mejor = [k for k, v in resultados.items() if v['modelo'] == mejor_modelo][0]
            print(f"\n🏆 Mejor modelo: {nombre_mejor}")
            
            y_pred = resultados[nombre_mejor]['predictions']
            y_test = resultados[nombre_mejor]['y_test']
            
            print("\n📊 Reporte de clasificación:")
            print(classification_report(y_test, y_pred, target_names=['No-Mosquito', 'Mosquito']))
            
            print("\n📊 Matriz de confusión:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
        
        return modelo_filename
    
    def cargar_modelo(self, archivo_modelo):
        """
        Carga un modelo pre-entrenado
        """
        try:
            with open(archivo_modelo, 'rb') as f:
                datos_modelo = pickle.load(f)
            
            self.modelo = datos_modelo['modelo']
            self.scaler = datos_modelo['scaler']
            self.modelo_entrenado = True
            
            print(f"✅ Modelo cargado desde: {archivo_modelo}")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def predecir(self, caracteristicas):
        """
        Realiza predicción usando el modelo entrenado
        """
        if not self.modelo_entrenado:
            raise Exception("Modelo no está entrenado")
        
        caracteristicas_scaled = self.scaler.transform([caracteristicas])
        probabilidad = self.modelo.predict_proba(caracteristicas_scaled)[0]
        prediccion = self.modelo.predict(caracteristicas_scaled)[0]
        
        return {
            'es_mosquito': prediccion == 1,
            'confianza': max(probabilidad),
            'probabilidad_mosquito': probabilidad[1] if len(probabilidad) > 1 else 0
        }
    
    def iniciar_camara(self, camara_id=0):
        """Inicia la cámara"""
        self.cap = cv2.VideoCapture(camara_id)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def deteccion_tiempo_real(self):
        """
        Ejecuta detección en tiempo real usando el modelo ML
        """
        if not self.modelo_entrenado:
            raise Exception("Modelo no está entrenado")
        
        if self.cap is None:
            raise Exception("Cámara no inicializada")
        
        print("🦟 Detector ML de mosquitos iniciado...")
        print("Controles:")
        print("  'q' - Salir")
        print("  's' - Capturar imagen")
        print("  'r' - Resetear estadísticas")
        print("  'c' - Mostrar estadísticas")
        
        cv2.namedWindow('Detector ML Mosquitos', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = time.time()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Procesar frame
                fg_mask = self.background_subtractor.apply(frame_gray)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                
                # Encontrar contornos
                contornos, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                mosquitos_detectados = []
                
                for contorno in contornos:
                    area = cv2.contourArea(contorno)
                    if self.MIN_AREA < area < self.MAX_AREA:
                        # Calcular centro para tracking
                        M = cv2.moments(contorno)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Calcular características de movimiento
                            movimiento_info = self.calcular_movimiento((cx, cy), timestamp)
                            
                            # Extraer características
                            caracteristicas = self.extraer_caracteristicas(contorno, frame_gray, movimiento_info)
                            
                            if caracteristicas is not None:
                                # Realizar predicción
                                prediccion = self.predecir(caracteristicas)
                                
                                if (prediccion['es_mosquito'] and 
                                    prediccion['confianza'] > self.detection_confidence_threshold):
                                    
                                    mosquitos_detectados.append({
                                        'contorno': contorno,
                                        'centro': (cx, cy),
                                        'confianza': prediccion['confianza'],
                                        'probabilidad': prediccion['probabilidad_mosquito'],
                                        'area': area
                                    })
                
                # Dibujar detecciones
                frame_display = frame.copy()
                
                for mosquito in mosquitos_detectados:
                    contorno = mosquito['contorno']
                    centro = mosquito['centro']
                    confianza = mosquito['confianza']
                    
                    # Color basado en confianza
                    color = (0, int(255 * confianza), int(255 * (1 - confianza)))
                    
                    cv2.drawContours(frame_display, [contorno], -1, color, 2)
                    cv2.circle(frame_display, centro, 5, (0, 0, 255), -1)
                    
                    # Mostrar confianza
                    texto = f"🦟 {confianza:.2f}"
                    cv2.putText(frame_display, texto, (centro[0] + 10, centro[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Actualizar estadísticas
                if mosquitos_detectados:
                    self.stats['total_detections'] += len(mosquitos_detectados)
                    self.stats['confirmed_mosquitos'] += len(mosquitos_detectados)
                    self.stats['detection_times'].append(timestamp)
                
                # Mostrar información
                fps = frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0
                info_lines = [
                    f"FPS: {fps:.1f} | Frame: {frame_count}",
                    f"Mosquitos detectados: {len(mosquitos_detectados)}",
                    f"Total: {self.stats['confirmed_mosquitos']}",
                    f"Umbral confianza: {self.detection_confidence_threshold:.2f}"
                ]
                
                for i, line in enumerate(info_lines):
                    y_pos = 25 + (i * 20)
                    cv2.putText(frame_display, line, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('Detector ML Mosquitos', frame_display)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"deteccion_ml_{timestamp_str}.jpg"
                    cv2.imwrite(filename, frame_display)
                    print(f"📷 Imagen guardada: {filename}")
                elif key == ord('r'):
                    self.stats = {
                        'total_detections': 0,
                        'confirmed_mosquitos': 0,
                        'false_positives': 0,
                        'detection_times': []
                    }
                    print("🔄 Estadísticas reseteadas")
                elif key == ord('c'):
                    self.mostrar_estadisticas()
        
        except KeyboardInterrupt:
            print("\n🛑 Detección interrumpida")
        
        finally:
            cv2.destroyAllWindows()
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas de detección"""
        print("\n📊 ESTADÍSTICAS DE DETECCIÓN:")
        print(f"  Total detecciones: {self.stats['total_detections']}")
        print(f"  Mosquitos confirmados: {self.stats['confirmed_mosquitos']}")
        print(f"  Falsos positivos: {self.stats['false_positives']}")
        print(f"  Umbral confianza: {self.detection_confidence_threshold:.2f}")
        
        if self.stats['detection_times']:
            tiempos = np.array(self.stats['detection_times'])
            duracion = tiempos[-1] - tiempos[0] if len(tiempos) > 1 else 0
            if duracion > 0:
                frecuencia = len(tiempos) / (duracion / 60)  # detecciones por minuto
                print(f"  Frecuencia: {frecuencia:.2f} detecciones/minuto")
    
    def ajustar_umbral_confianza(self, nuevo_umbral):
        """Ajusta el umbral de confianza para detección"""
        self.detection_confidence_threshold = max(0.0, min(1.0, nuevo_umbral))
        print(f"🎚️ Umbral de confianza ajustado a: {self.detection_confidence_threshold:.2f}")

def main():
    detector = DetectorMosquitoML()
    
    print("🦟 DETECTOR DE MOSQUITOS CON MACHINE LEARNING")
    print("=" * 50)
    
    try:
        detector.iniciar_camara(0)
        
        while True:
            print("\n📋 MENÚ PRINCIPAL:")
            print("1. Recolectar datos de entrenamiento")
            print("2. Entrenar nuevo modelo")
            print("3. Cargar modelo existente")
            print("4. Detección en tiempo real")
            print("5. Ajustar umbral de confianza")
            print("6. Mostrar estadísticas")
            print("7. Salir")
            
            opcion = input("\nSelecciona una opción (1-7): ").strip()
            
            if opcion == "1":
                minutos = input("¿Cuántos minutos recolectar datos? (default: 5): ").strip()
                minutos = int(minutos) if minutos.isdigit() else 5
                
                datos = detector.recolectar_datos_entrenamiento(minutos)
                
                if datos:
                    entrenar = input("¿Entrenar modelo con estos datos? (s/n): ").strip().lower()
                    if entrenar == 's':
                        detector.entrenar_modelo(datos_dict=datos)
            
            elif opcion == "2":
                archivos_datos = [f for f in os.listdir('.') if f.startswith('datos_mosquitos_') and f.endswith('.pkl')]
                
                if not archivos_datos:
                    print("❌ No se encontraron archivos de datos. Recolecta datos primero.")
                    continue
                
                print("\n📁 Archivos de datos disponibles:")
                for i, archivo in enumerate(archivos_datos):
                    print(f"  {i+1}. {archivo}")
                
                seleccion = input("Selecciona archivo (número): ").strip()
                if seleccion.isdigit() and 1 <= int(seleccion) <= len(archivos_datos):
                    archivo_seleccionado = archivos_datos[int(seleccion)-1]
                    detector.entrenar_modelo(datos_archivo=archivo_seleccionado)
                else:
                    print("❌ Selección inválida")
            
            elif opcion == "3":
                archivos_modelo = [f for f in os.listdir('.') if f.startswith('modelo_mosquito_') and f.endswith('.pkl')]
                
                if not archivos_modelo:
                    print("❌ No se encontraron modelos entrenados. Entrena un modelo primero.")
                    continue
                
                print("\n🤖 Modelos disponibles:")
                for i, archivo in enumerate(archivos_modelo):
                    print(f"  {i+1}. {archivo}")
                
                seleccion = input("Selecciona modelo (número): ").strip()
                if seleccion.isdigit() and 1 <= int(seleccion) <= len(archivos_modelo):
                    archivo_seleccionado = archivos_modelo[int(seleccion)-1]
                    detector.cargar_modelo(archivo_seleccionado)
                else:
                    print("❌ Selección inválida")
            
            elif opcion == "4":
                if not detector.modelo_entrenado:
                    print("❌ No hay modelo cargado. Carga o entrena un modelo primero.")
                    continue
                
                detector.deteccion_tiempo_real()
            
            elif opcion == "5":
                umbral_actual = detector.detection_confidence_threshold
                print(f"Umbral actual: {umbral_actual:.2f}")
                
                nuevo_umbral = input("Nuevo umbral (0.0-1.0): ").strip()
                try:
                    detector.ajustar_umbral_confianza(float(nuevo_umbral))
                except ValueError:
                    print("❌ Valor inválido")
            
            elif opcion == "6":
                detector.mostrar_estadisticas()
            
            elif opcion == "7":
                print("👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción inválida")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        if detector.cap:
            detector.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()