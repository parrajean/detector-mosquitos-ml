import numpy as np
import os
import cv2
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, classification_report
from collections import deque, Counter
import threading

class DetectorMosquitoMejorado:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.modelo_entrenado = False
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Configuración de cámara mejorada
        self.cap = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=25, history=200)
        
        # Parámetros más restrictivos
        self.MIN_AREA = 15          # Aumentado de 5
        self.MAX_AREA = 300         # Reducido de 500
        self.MIN_PERIMETER = 15     # Nuevo filtro
        self.MAX_PERIMETER = 100    # Nuevo filtro
        
        # Sistema de confianza mejorado
        self.detection_confidence_threshold = 0.85  # Aumentado de 0.7
        self.min_consecutive_detections = 3         # Nuevo: mínimas detecciones consecutivas
        self.temporal_consistency_window = 10       # Ventana para consistencia temporal
        
        # Tracking avanzado para reducir falsos positivos
        self.active_tracks = {}
        self.next_track_id = 0
        self.max_track_distance = 50
        self.min_track_frames = 5    # Mínimos frames para confirmar
        
        # Filtros adicionales
        self.movement_threshold = 5.0     # Mínimo movimiento requerido
        self.shape_consistency_threshold = 0.3  # Consistencia de forma
        
        # Estadísticas mejoradas
        self.stats = {
            'total_detections': 0,
            'confirmed_mosquitos': 0,
            'false_positives': 0,
            'filtered_out': 0,
            'precision_score': 0.0
        }
        
        # Buffer para validación temporal
        self.detection_buffer = deque(maxlen=30)
        
    def extraer_caracteristicas_mejoradas(self, contorno, frame_gray, movimiento_info=None, frame_original=None):
        """Extrae características más discriminativas"""
        caracteristicas = []
        
        # 1. Filtros básicos mejorados
        area = cv2.contourArea(contorno)
        if area < self.MIN_AREA or area > self.MAX_AREA:
            return None
            
        perimetro = cv2.arcLength(contorno, True)
        if perimetro < self.MIN_PERIMETER or perimetro > self.MAX_PERIMETER:
            return None
        
        # 2. Características geométricas más estrictas
        area_perimetro_ratio = area / (perimetro ** 2) if perimetro > 0 else 0
        caracteristicas.extend([area, perimetro, area_perimetro_ratio])
        
        # 3. Rectángulo delimitador con validaciones
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / h if h > 0 else 0
        
        # Filtros de forma típicos de mosquitos
        if aspect_ratio < 1.2 or aspect_ratio > 5.0:  # Mosquitos son alargados
            return None
            
        extent = area / (w * h) if (w * h) > 0 else 0
        if extent < 0.3:  # Muy disperso, probablemente ruido
            return None
            
        caracteristicas.extend([w, h, aspect_ratio, extent])
        
        # 4. Análisis de forma más detallado
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Mosquitos tienen solidity característico
        if solidity < 0.4 or solidity > 0.95:
            return None
            
        compactness = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
        caracteristicas.extend([solidity, compactness])
        
        # 5. Momentos de Hu mejorados
        momentos = cv2.moments(contorno)
        if momentos['m00'] != 0:
            hu_moments = cv2.HuMoments(momentos)
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            # Filtrar formas muy irregulares
            if np.any(np.abs(hu_moments[:2]) > 5):  # Formas muy asimétricas
                return None
                
            caracteristicas.extend(hu_moments.flatten())
        else:
            return None
        
        # 6. Análisis de textura mejorado
        mask = np.zeros(frame_gray.shape, np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)
        
        # Región de interés expandida para mejor análisis
        x_exp = max(0, x-5)
        y_exp = max(0, y-5)
        w_exp = min(frame_gray.shape[1] - x_exp, w + 10)
        h_exp = min(frame_gray.shape[0] - y_exp, h + 10)
        
        roi_mask = mask[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
        roi_gray = frame_gray[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
        
        if roi_mask.size > 0 and roi_gray.size > 0:
            intensidades = roi_gray[roi_mask[0:roi_gray.shape[0], 0:roi_gray.shape[1]] == 255]
            
            if len(intensidades) > 5:  # Suficientes píxeles para análisis
                mean_intensity = np.mean(intensidades)
                std_intensity = np.std(intensidades)
                
                # Análisis de gradiente (bordes)
                grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                mean_gradient = np.mean(gradient_magnitude)
                
                caracteristicas.extend([mean_intensity, std_intensity, mean_gradient])
            else:
                return None
        else:
            return None
        
        # 7. Características de movimiento validadas
        if movimiento_info:
            velocidad = movimiento_info.get('velocidad', 0)
            aceleracion = movimiento_info.get('aceleracion', 0)
            cambios_direccion = movimiento_info.get('cambios_direccion', 0)
            
            # Filtros de movimiento (mosquitos se mueven característicamente)
            if velocidad < self.movement_threshold:  # Muy poco movimiento
                return None
                
            if cambios_direccion < 1:  # Mosquitos cambian dirección frecuentemente
                return None
            
            # Análisis de patrón de vuelo
            patron_vuelo = self.analizar_patron_vuelo(movimiento_info)
            caracteristicas.extend([velocidad, aceleracion, cambios_direccion, patron_vuelo])
        else:
            caracteristicas.extend([0, 0, 0, 0])
        
        # 8. Características adicionales específicas
        elongacion = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        densidad = np.sum(mask == 255) / (w * h) if (w * h) > 0 else 0
        
        # Factor de forma circular (mosquitos no son muy circulares)
        circularidad = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
        if circularidad > 0.8:  # Demasiado circular, probablemente no es mosquito
            return None
        
        caracteristicas.extend([elongacion, densidad, circularidad])
        
        return np.array(caracteristicas)
    
    def analizar_patron_vuelo(self, movimiento_info):
        """Analiza si el patrón de movimiento es típico de mosquitos"""
        velocidad = movimiento_info.get('velocidad', 0)
        cambios_direccion = movimiento_info.get('cambios_direccion', 0)
        
        # Patrón típico: velocidad moderada con muchos cambios de dirección
        if velocidad > 20 and velocidad < 150 and cambios_direccion >= 2:
            return 1.0  # Patrón típico de mosquito
        elif velocidad > 150:  # Muy rápido, probablemente no es mosquito
            return 0.2
        elif cambios_direccion == 0:  # Sin cambios, movimiento lineal
            return 0.1
        else:
            return 0.5
    
    def calcular_movimiento_mejorado(self, centro_actual, timestamp, track_id=None):
        """Calcula características de movimiento con mejor tracking"""
        if track_id is None:
            return {'velocidad': 0, 'aceleracion': 0, 'cambios_direccion': 0}
        
        if track_id not in self.active_tracks:
            self.active_tracks[track_id] = {
                'positions': deque(maxlen=15),
                'timestamps': deque(maxlen=15),
                'velocidades': deque(maxlen=10)
            }
        
        track = self.active_tracks[track_id]
        track['positions'].append(centro_actual)
        track['timestamps'].append(timestamp)
        
        if len(track['positions']) < 2:
            return {'velocidad': 0, 'aceleracion': 0, 'cambios_direccion': 0}
        
        # Calcular velocidades instantáneas
        velocidades = []
        for i in range(1, len(track['positions'])):
            p1 = track['positions'][i-1]
            p2 = track['positions'][i]
            t1 = track['timestamps'][i-1]
            t2 = track['timestamps'][i]
            
            if t2 > t1:
                distancia = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                velocidad = distancia / (t2 - t1)
                velocidades.append(velocidad)
        
        if not velocidades:
            return {'velocidad': 0, 'aceleracion': 0, 'cambios_direccion': 0}
        
        velocidad_promedio = np.mean(velocidades)
        track['velocidades'].extend(velocidades)
        
        # Calcular aceleración como variabilidad de velocidad
        aceleracion = np.std(list(track['velocidades'])) if len(track['velocidades']) > 1 else 0
        
        # Contar cambios de dirección significativos
        cambios_direccion = self.contar_cambios_direccion(list(track['positions']))
        
        return {
            'velocidad': velocidad_promedio,
            'aceleracion': aceleracion,
            'cambios_direccion': cambios_direccion
        }
    
    def contar_cambios_direccion(self, positions):
        """Cuenta cambios de dirección significativos"""
        if len(positions) < 3:
            return 0
        
        cambios = 0
        umbral_angulo = np.pi / 3  # 60 grados
        
        for i in range(2, len(positions)):
            p1 = positions[i-2]
            p2 = positions[i-1]
            p3 = positions[i]
            
            # Vectores de dirección
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Evitar división por cero
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1 and norm2 > 1:  # Solo si hay movimiento significativo
                cos_angulo = np.dot(v1, v2) / (norm1 * norm2)
                cos_angulo = np.clip(cos_angulo, -1, 1)  # Evitar errores numéricos
                
                angulo = np.arccos(cos_angulo)
                if angulo > umbral_angulo:
                    cambios += 1
        
        return cambios
    
    def sistema_tracking_mejorado(self, detecciones_frame, timestamp):
        """Sistema de tracking para validar detecciones temporales"""
        detecciones_validadas = []
        
        # Asociar detecciones con tracks existentes
        for deteccion in detecciones_frame:
            centro = deteccion['centro']
            mejor_track = None
            menor_distancia = float('inf')
            
            # Buscar el track más cercano
            for track_id, track_info in self.active_tracks.items():
                if len(track_info['positions']) > 0:
                    ultima_pos = track_info['positions'][-1]
                    distancia = np.sqrt((centro[0] - ultima_pos[0])**2 + 
                                      (centro[1] - ultima_pos[1])**2)
                    
                    if distancia < self.max_track_distance and distancia < menor_distancia:
                        mejor_track = track_id
                        menor_distancia = distancia
            
            # Asignar a track existente o crear nuevo
            if mejor_track is not None:
                track_id = mejor_track
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[track_id] = {
                    'positions': deque(maxlen=15),
                    'timestamps': deque(maxlen=15),
                    'velocidades': deque(maxlen=10),
                    'confirmations': 0,
                    'consecutive_detections': 0
                }
            
            # Actualizar track
            track = self.active_tracks[track_id]
            track['positions'].append(centro)
            track['timestamps'].append(timestamp)
            track['consecutive_detections'] += 1
            
            # Solo confirmar si cumple criterios temporales
            if track['consecutive_detections'] >= self.min_consecutive_detections:
                deteccion['track_id'] = track_id
                deteccion['validated'] = True
                detecciones_validadas.append(deteccion)
            else:
                deteccion['track_id'] = track_id
                deteccion['validated'] = False
        
        # Limpiar tracks antiguos
        tracks_a_eliminar = []
        for track_id, track_info in self.active_tracks.items():
            if len(track_info['timestamps']) > 0:
                tiempo_desde_ultima = timestamp - track_info['timestamps'][-1]
                if tiempo_desde_ultima > 2.0:  # 2 segundos sin detección
                    tracks_a_eliminar.append(track_id)
        
        for track_id in tracks_a_eliminar:
            del self.active_tracks[track_id]
        
        return detecciones_validadas
    
    def validacion_ensemble(self, caracteristicas):
        """Sistema de validación con múltiples modelos"""
        if not self.modelo_entrenado:
            return {'es_mosquito': False, 'confianza': 0.0}
        
        caracteristicas_scaled = self.scaler.transform([caracteristicas])
        
        # Predicción del modelo principal
        probabilidad = self.modelo.predict_proba(caracteristicas_scaled)[0]
        prediccion_principal = probabilidad[1] if len(probabilidad) > 1 else 0
        
        # Detección de anomalías (para filtrar outliers)
        anomalia = self.anomaly_detector.predict(caracteristicas_scaled)[0]
        if anomalia == -1:  # Es una anomalía
            return {'es_mosquito': False, 'confianza': 0.0}
        
        # Sistema de validación por rangos (conocimiento de dominio)
        validacion_rangos = self.validar_por_rangos(caracteristicas)
        
        # Combinar todas las validaciones
        confianza_final = prediccion_principal * validacion_rangos
        
        return {
            'es_mosquito': confianza_final > self.detection_confidence_threshold,
            'confianza': confianza_final,
            'prediccion_ml': prediccion_principal,
            'validacion_rangos': validacion_rangos
        }
    
    def validar_por_rangos(self, caracteristicas):
        """Validación basada en conocimiento de dominio"""
        try:
            area = caracteristicas[0]
            aspect_ratio = caracteristicas[5]
            solidity = caracteristicas[7]
            velocidad = caracteristicas[19] if len(caracteristicas) > 19 else 0
            elongacion = caracteristicas[-3] if len(caracteristicas) > 22 else 0
            
            score = 1.0
            
            # Validaciones de rango típico de mosquitos
            if not (20 <= area <= 200):
                score *= 0.5
            
            if not (1.5 <= aspect_ratio <= 4.0):
                score *= 0.3
            
            if not (0.4 <= solidity <= 0.85):
                score *= 0.4
            
            if velocidad > 0 and not (10 <= velocidad <= 120):
                score *= 0.6
            
            if not (1.8 <= elongacion <= 4.5):
                score *= 0.5
            
            return max(0.1, score)  # Mínimo 0.1
            
        except (IndexError, ValueError):
            return 0.5  # Valor neutral si hay error
    
    def deteccion_tiempo_real_mejorada(self):
        """Detección en tiempo real con filtros mejorados"""
        if not self.modelo_entrenado:
            raise Exception("Modelo no está entrenado")
        
        if self.cap is None:
            raise Exception("Cámara no inicializada")
        
        print("Detector ML mejorado iniciado...")
        print("Controles:")
        print("  'q' - Salir")
        print("  'f' - Marcar como falso positivo")
        print("  'v' - Marcar como verdadero positivo")
        print("  's' - Capturar imagen")
        print("  'r' - Resetear estadísticas")
        
        cv2.namedWindow('Detector Mejorado', cv2.WINDOW_NORMAL)
        
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
                
                # Procesamiento con parámetros más conservadores
                fg_mask = self.background_subtractor.apply(frame_gray)
                
                # Filtrado morfológico más agresivo
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                
                # Filtrado adicional por área en la máscara
                contornos_mask, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                fg_mask_filtrada = np.zeros_like(fg_mask)
                
                for contorno in contornos_mask:
                    area = cv2.contourArea(contorno)
                    if self.MIN_AREA <= area <= self.MAX_AREA:
                        cv2.drawContours(fg_mask_filtrada, [contorno], -1, 255, -1)
                
                # Encontrar contornos finales
                contornos, _ = cv2.findContours(fg_mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Procesar cada contorno con filtros estrictos
                detecciones_frame = []
                
                for contorno in contornos:
                    # Pre-filtros rápidos
                    area = cv2.contourArea(contorno)
                    if not (self.MIN_AREA <= area <= self.MAX_AREA):
                        continue
                    
                    perimetro = cv2.arcLength(contorno, True)
                    if not (self.MIN_PERIMETER <= perimetro <= self.MAX_PERIMETER):
                        continue
                    
                    # Calcular centro
                    M = cv2.moments(contorno)
                    if M["m00"] == 0:
                        continue
                        
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calcular movimiento con tracking temporal
                    movimiento_info = self.calcular_movimiento_mejorado((cx, cy), timestamp, f"temp_{cx}_{cy}")
                    
                    # Extraer características con filtros estrictos
                    caracteristicas = self.extraer_caracteristicas_mejoradas(
                        contorno, frame_gray, movimiento_info, frame
                    )
                    
                    if caracteristicas is not None:
                        # Validación con ensemble
                        resultado = self.validacion_ensemble(caracteristicas)
                        
                        if resultado['es_mosquito']:
                            detecciones_frame.append({
                                'contorno': contorno,
                                'centro': (cx, cy),
                                'confianza': resultado['confianza'],
                                'area': area,
                                'caracteristicas': caracteristicas,
                                'resultado_completo': resultado
                            })
                        else:
                            self.stats['filtered_out'] += 1
                
                # Sistema de tracking temporal
                detecciones_validadas = self.sistema_tracking_mejorado(detecciones_frame, timestamp)
                
                # Mostrar solo detecciones validadas
                frame_display = frame.copy()
                mosquitos_confirmados = 0
                
                for deteccion in detecciones_validadas:
                    if deteccion.get('validated', False):
                        contorno = deteccion['contorno']
                        centro = deteccion['centro']
                        confianza = deteccion['confianza']
                        
                        # Color basado en confianza
                        if confianza > 0.9:
                            color = (0, 255, 0)  # Verde: muy confiable
                        elif confianza > 0.8:
                            color = (0, 255, 255)  # Amarillo: confiable
                        else:
                            color = (0, 165, 255)  # Naranja: menos confiable
                        
                        cv2.drawContours(frame_display, [contorno], -1, color, 2)
                        cv2.circle(frame_display, centro, 3, (0, 0, 255), -1)
                        
                        # Mostrar información detallada
                        texto = f"M:{confianza:.2f}"
                        cv2.putText(frame_display, texto, (centro[0] + 5, centro[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        mosquitos_confirmados += 1
                
                # Actualizar estadísticas
                if mosquitos_confirmados > 0:
                    self.stats['confirmed_mosquitos'] += mosquitos_confirmados
                    self.stats['total_detections'] += mosquitos_confirmados
                
                # Mostrar información mejorada
                fps = frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0
                
                info_lines = [
                    f"FPS: {fps:.1f} | Frame: {frame_count}",
                    f"Mosquitos confirmados: {mosquitos_confirmados}",
                    f"Total confirmados: {self.stats['confirmed_mosquitos']}",
                    f"Filtrados: {self.stats['filtered_out']}",
                    f"Tracks activos: {len(self.active_tracks)}",
                    f"Umbral: {self.detection_confidence_threshold:.2f}"
                ]
                
                for i, line in enumerate(info_lines):
                    y_pos = 20 + (i * 15)
                    cv2.putText(frame_display, line, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.imshow('Detector Mejorado', frame_display)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Marcar último como falso positivo
                    if mosquitos_confirmados > 0:
                        self.stats['false_positives'] += 1
                        self.stats['confirmed_mosquitos'] -= 1
                        print(f"Marcado como falso positivo. FP: {self.stats['false_positives']}")
                elif key == ord('v'):
                    # Confirmar como verdadero positivo
                    print(f"Confirmado como verdadero positivo")
                elif key == ord('s'):
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"deteccion_mejorada_{timestamp_str}.jpg"
                    cv2.imwrite(filename, frame_display)
                    print(f"Imagen guardada: {filename}")
                elif key == ord('r'):
                    self.stats = {
                        'total_detections': 0,
                        'confirmed_mosquitos': 0,
                        'false_positives': 0,
                        'filtered_out': 0,
                        'precision_score': 0.0
                    }
                    print("Estadísticas reseteadas")
        
        except KeyboardInterrupt:
            print("\nDetección interrumpida")
        
        finally:
            # Calcular precisión final
            total = self.stats['confirmed_mosquitos'] + self.stats['false_positives']
            if total > 0:
                precision = self.stats['confirmed_mosquitos'] / total
                self.stats['precision_score'] = precision
                print(f"\nPrecisión estimada: {precision:.2%}")
                print(f"Verdaderos positivos: {self.stats['confirmed_mosquitos']}")
                print(f"Falsos positivos: {self.stats['false_positives']}")
                print(f"Objetos filtrados: {self.stats['filtered_out']}")
            
            cv2.destroyAllWindows()
    
    def entrenar_con_datos_balanceados(self, X, y):
        """Entrena el modelo con técnicas para manejar desbalance"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calcular pesos para clases desbalanceadas
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Pesos de clases: {class_weight_dict}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo principal con pesos balanceados
        self.modelo = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        self.modelo.fit(X_train_scaled, y_train)
        
        # Entrenar detector de anomalías con datos negativos
        X_negative = X_train_scaled[y_train == 0]
        if len(X_negative) > 10:
            self.anomaly_detector.fit(X_negative)
        
        self.modelo_entrenado = True
        
        # Evaluar y ajustar umbral
        if len(X_test) > 0:
            y_proba = self.modelo.predict_proba(X_test_scaled)[:, 1]
            self.ajustar_umbral_optimo(y_test, y_proba)
        
        return True
    
    def ajustar_umbral_optimo(self, y_true, y_proba):
        """Encuentra el umbral óptimo para maximizar precisión"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Buscar umbral que maximice F1-score con sesgo hacia precisión
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Dar más peso a la precisión para reducir falsos positivos
        weighted_scores = 0.7 * precision + 0.3 * recall
        
        best_threshold_idx = np.argmax(weighted_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        # Asegurar que el umbral no sea demasiado bajo
        self.detection_confidence_threshold = max(0.75, min(0.95, best_threshold))
        
        print(f"Umbral óptimo ajustado a: {self.detection_confidence_threshold:.3f}")
        print(f"Precisión esperada: {precision[best_threshold_idx]:.3f}")
        print(f"Recall esperado: {recall[best_threshold_idx]:.3f}")
    
    def iniciar_camara(self, camara_id=0):
        """Inicia la cámara con configuración optimizada"""
        self.cap = cv2.VideoCapture(camara_id)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara")
        
        # Configuración optimizada para detección de objetos pequeños
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Reducir exposición para mejor detección
        
        print("Cámara configurada con parámetros optimizados")
    
    def calibrar_fondo(self, duracion_segundos=10):
        """Calibra el detector de fondo para reducir falsos positivos"""
        if self.cap is None:
            raise Exception("Cámara no inicializada")
        
        print(f"Calibrando fondo por {duracion_segundos} segundos...")
        print("Mantén el área de detección sin mosquitos...")
        
        fin_tiempo = time.time() + duracion_segundos
        frame_count = 0
        
        while time.time() < fin_tiempo:
            ret, frame = self.cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.background_subtractor.apply(frame_gray, learningRate=0.01)
                frame_count += 1
                
                # Mostrar progreso
                tiempo_restante = fin_tiempo - time.time()
                cv2.putText(frame, f"Calibrando: {tiempo_restante:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Calibración', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        print(f"Calibración completada con {frame_count} frames")
    
    def modo_entrenamiento_interactivo(self):
        """Modo especial para entrenar con retroalimentación del usuario"""
        if self.cap is None:
            raise Exception("Cámara no inicializada")
        
        print("MODO ENTRENAMIENTO INTERACTIVO")
        print("Controles:")
        print("  'm' - Marcar detección como MOSQUITO")
        print("  'n' - Marcar detección como NO-MOSQUITO") 
        print("  'i' - Ignorar detección actual")
        print("  'q' - Guardar y salir")
        
        datos_entrenamiento = []
        etiquetas = []
        
        cv2.namedWindow('Entrenamiento Interactivo', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = self.background_subtractor.apply(frame_gray)
                
                # Procesamiento básico
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                
                contornos, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Mostrar detecciones para etiquetar
                frame_display = frame.copy()
                detecciones_disponibles = []
                
                for i, contorno in enumerate(contornos):
                    area = cv2.contourArea(contorno)
                    if self.MIN_AREA <= area <= self.MAX_AREA:
                        cv2.drawContours(frame_display, [contorno], -1, (0, 255, 255), 2)
                        
                        M = cv2.moments(contorno)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(frame_display, str(i), (cx, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Extraer características para entrenamiento
                            caracteristicas = self.extraer_caracteristicas_mejoradas(
                                contorno, frame_gray, None, frame
                            )
                            
                            if caracteristicas is not None:
                                detecciones_disponibles.append({
                                    'contorno': contorno,
                                    'caracteristicas': caracteristicas,
                                    'indice': i
                                })
                
                # Mostrar información
                info_text = f"Detecciones: {len(detecciones_disponibles)} | Muestras: {len(datos_entrenamiento)}"
                cv2.putText(frame_display, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Entrenamiento Interactivo', frame_display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('m') and detecciones_disponibles:
                    # Marcar todas las detecciones actuales como mosquitos
                    for det in detecciones_disponibles:
                        datos_entrenamiento.append(det['caracteristicas'])
                        etiquetas.append(1)
                    print(f"✅ {len(detecciones_disponibles)} muestras marcadas como MOSQUITOS")
                    
                elif key == ord('n') and detecciones_disponibles:
                    # Marcar todas las detecciones actuales como no-mosquitos
                    for det in detecciones_disponibles:
                        datos_entrenamiento.append(det['caracteristicas'])
                        etiquetas.append(0)
                    print(f"❌ {len(detecciones_disponibles)} muestras marcadas como NO-MOSQUITOS")
        
        except KeyboardInterrupt:
            pass
        
        finally:
            cv2.destroyAllWindows()
            
            if len(datos_entrenamiento) > 0:
                # Entrenar con datos recolectados
                X = np.array(datos_entrenamiento)
                y = np.array(etiquetas)
                
                print(f"\n📊 Entrenando con {len(X)} muestras...")
                print(f"Mosquitos: {sum(y)}, No-mosquitos: {len(y) - sum(y)}")
                
                self.entrenar_con_datos_balanceados(X, y)
                
                # Guardar datos
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"datos_interactivos_{timestamp}.pkl"
                
                with open(filename, 'wb') as f:
                    pickle.dump({
                        'caracteristicas': X,
                        'etiquetas': y,
                        'timestamp': time.time()
                    }, f)
                
                print(f"Datos guardados en: {filename}")
                return True
            else:
                print("No se recolectaron datos de entrenamiento")
                return False

# Clase principal mejorada que reemplaza la original
class DetectorMosquitoMLMejorado(DetectorMosquitoMejorado):
    """Versión mejorada del detector con todas las mejoras implementadas"""
    
    def __init__(self):
        super().__init__()
        print("Detector ML Mejorado inicializado")
        print("Mejoras implementadas:")
        print("  ✅ Filtros geométricos más estrictos")
        print("  ✅ Análisis de movimiento avanzado") 
        print("  ✅ Sistema de tracking temporal")
        print("  ✅ Validación por ensemble")
        print("  ✅ Detección de anomalías")
        print("  ✅ Umbral de confianza adaptativo")
    
    def menu_principal(self):
        """Menú principal con opciones mejoradas"""
        while True:
            print("\n" + "="*50)
            print("🦟 DETECTOR ML DE MOSQUITOS - VERSIÓN MEJORADA")
            print("="*50)
            
            print("\n📋 MENÚ PRINCIPAL:")
            print("1. Calibrar fondo (recomendado antes de usar)")
            print("2. Entrenamiento interactivo con cámara")
            print("3. Cargar modelo existente")
            print("4. Detección en tiempo real (mejorada)")
            print("5. Ajustar parámetros de detección")
            print("6. Mostrar estadísticas")
            print("7. Modo diagnóstico")
            print("8. Salir")
            
            opcion = input("\nSelecciona una opción (1-8): ").strip()
            
            try:
                if opcion == "1":
                    if self.cap is None:
                        self.iniciar_camara(0)
                    
                    duracion = input("Duración de calibración en segundos (default: 10): ").strip()
                    duracion = int(duracion) if duracion.isdigit() else 10
                    self.calibrar_fondo(duracion)
                
                elif opcion == "2":
                    if self.cap is None:
                        self.iniciar_camara(0)
                    
                    if self.modo_entrenamiento_interactivo():
                        print("✅ Modelo entrenado con datos interactivos")
                
                elif opcion == "3":
                    archivos_modelo = [f for f in os.listdir('.') 
                                     if f.startswith('modelo_mosquito_') and f.endswith('.pkl')]
                    
                    if not archivos_modelo:
                        print("❌ No se encontraron modelos entrenados")
                        continue
                    
                    print("\n🤖 Modelos disponibles:")
                    for i, archivo in enumerate(archivos_modelo):
                        print(f"  {i+1}. {archivo}")
                    
                    seleccion = input("Selecciona modelo (número): ").strip()
                    if seleccion.isdigit() and 1 <= int(seleccion) <= len(archivos_modelo):
                        archivo_seleccionado = archivos_modelo[int(seleccion)-1]
                        
                        with open(archivo_seleccionado, 'rb') as f:
                            datos_modelo = pickle.load(f)
                        
                        self.modelo = datos_modelo['modelo']
                        self.scaler = datos_modelo['scaler']
                        self.modelo_entrenado = True
                        
                        print(f"✅ Modelo cargado: {archivo_seleccionado}")
                
                elif opcion == "4":
                    if not self.modelo_entrenado:
                        print("❌ No hay modelo cargado. Entrena o carga un modelo primero.")
                        continue
                    
                    if self.cap is None:
                        self.iniciar_camara(0)
                    
                    self.deteccion_tiempo_real_mejorada()
                
                elif opcion == "5":
                    self.ajustar_parametros()
                
                elif opcion == "6":
                    self.mostrar_estadisticas_detalladas()
                
                elif opcion == "7":
                    self.modo_diagnostico()
                
                elif opcion == "8":
                    print("👋 ¡Hasta luego!")
                    break
                
                else:
                    print("❌ Opción inválida")
            
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def ajustar_parametros(self):
        """Interfaz para ajustar parámetros del detector"""
        print("\n⚙️ AJUSTE DE PARÁMETROS:")
        print(f"1. Área mínima (actual: {self.MIN_AREA})")
        print(f"2. Área máxima (actual: {self.MAX_AREA})")
        print(f"3. Umbral de confianza (actual: {self.detection_confidence_threshold:.2f})")
        print(f"4. Mínimas detecciones consecutivas (actual: {self.min_consecutive_detections})")
        print(f"5. Umbral de movimiento (actual: {self.movement_threshold})")
        
        while True:
            opcion = input("Selecciona parámetro a cambiar (1-5) o 'q' para continuar: ")
            
            if opcion == 'q':
                break
            elif opcion == '1':
                try:
                    nuevo_valor = float(input(f"Nueva área mínima (actual: {self.MIN_AREA}): "))
                    self.MIN_AREA = max(5, min(100, nuevo_valor))
                    print(f"✅ Área mínima: {self.MIN_AREA}")
                except ValueError:
                    print("❌ Valor inválido")
            elif opcion == '2':
                try:
                    nuevo_valor = float(input(f"Nueva área máxima (actual: {self.MAX_AREA}): "))
                    self.MAX_AREA = max(100, min(1000, nuevo_valor))
                    print(f"✅ Área máxima: {self.MAX_AREA}")
                except ValueError:
                    print("❌ Valor inválido")
            elif opcion == '3':
                try:
                    nuevo_valor = float(input(f"Nuevo umbral de confianza (0.5-0.99): "))
                    self.detection_confidence_threshold = max(0.5, min(0.99, nuevo_valor))
                    print(f"✅ Umbral de confianza: {self.detection_confidence_threshold:.2f}")
                except ValueError:
                    print("❌ Valor inválido")
            elif opcion == '4':
                try:
                    nuevo_valor = int(input(f"Mínimas detecciones consecutivas (1-10): "))
                    self.min_consecutive_detections = max(1, min(10, nuevo_valor))
                    print(f"✅ Mínimas detecciones consecutivas: {self.min_consecutive_detections}")
                except ValueError:
                    print("❌ Valor inválido")
            elif opcion == '5':
                try:
                    nuevo_valor = float(input(f"Umbral de movimiento (1-20): "))
                    self.movement_threshold = max(1, min(20, nuevo_valor))
                    print(f"✅ Umbral de movimiento: {self.movement_threshold}")
                except ValueError:
                    print("❌ Valor inválido")
    
    def mostrar_estadisticas_detalladas(self):
        """Muestra estadísticas detalladas del detector"""
        print("\n📊 ESTADÍSTICAS DETALLADAS:")
        print("=" * 40)
        print(f"Total detecciones: {self.stats['total_detections']}")
        print(f"Mosquitos confirmados: {self.stats['confirmed_mosquitos']}")
        print(f"Falsos positivos: {self.stats['false_positives']}")
        print(f"Objetos filtrados: {self.stats['filtered_out']}")
        
        # Calcular métricas
        total_clasificaciones = self.stats['confirmed_mosquitos'] + self.stats['false_positives']
        if total_clasificaciones > 0:
            precision = self.stats['confirmed_mosquitos'] / total_clasificaciones
            print(f"Precisión estimada: {precision:.2%}")
        
        total_objetos = total_clasificaciones + self.stats['filtered_out']
        if total_objetos > 0:
            tasa_filtrado = self.stats['filtered_out'] / total_objetos
            print(f"Tasa de filtrado: {tasa_filtrado:.2%}")
        
        print(f"Tracks activos: {len(self.active_tracks)}")
        print(f"Umbral actual: {self.detection_confidence_threshold:.3f}")
    
    def modo_diagnostico(self):
        """Modo diagnóstico para depuración"""
        if self.cap is None:
            self.iniciar_camara(0)
        
        print("\n🔧 MODO DIAGNÓSTICO")
        print("Mostrará información detallada de procesamiento")
        
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Máscara', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Contornos', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = self.background_subtractor.apply(frame_gray)
                
                # Mostrar pasos del procesamiento
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                fg_mask_procesada = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask_procesada = cv2.morphologyEx(fg_mask_procesada, cv2.MORPH_CLOSE, kernel)
                
                # Encontrar y analizar contornos
                contornos, _ = cv2.findContours(fg_mask_procesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                frame_contornos = frame.copy()
                info_contornos = []
                
                for i, contorno in enumerate(contornos):
                    area = cv2.contourArea(contorno)
                    perimetro = cv2.arcLength(contorno, True)
                    
                    if area > 1:  # Mostrar todos los contornos
                        color = (0, 255, 0) if self.MIN_AREA <= area <= self.MAX_AREA else (0, 0, 255)
                        cv2.drawContours(frame_contornos, [contorno], -1, color, 1)
                        
                        M = cv2.moments(contorno)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            texto = f"{i}: A={area:.0f}"
                            cv2.putText(frame_contornos, texto, (cx-20, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                            
                            info_contornos.append(f"Contorno {i}: Área={area:.1f}, Perímetro={perimetro:.1f}")
                
                # Mostrar información en consola
                if info_contornos:
                    print(f"\rContornos detectados: {len(contornos)}", end="")
                
                cv2.imshow('Original', frame)
                cv2.imshow('Máscara', fg_mask_procesada)
                cv2.imshow('Contornos', frame_contornos)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            cv2.destroyAllWindows()

def main():
    print("Iniciando Detector ML de Mosquitos Mejorado...")
    
    detector = DetectorMosquitoMLMejorado()
    
    try:
        detector.menu_principal()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector.cap:
            detector.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()