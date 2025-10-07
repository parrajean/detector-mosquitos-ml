from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
import time
import json
import threading
import os
from datetime import datetime
import base64
from detector_mejorado import DetectorMosquitoMLMejorado

# Configuración de Flask
app = Flask(__name__)
app.secret_key = 'mosquito_detector_ml_2024'

# Variables globales para control de estado
detector = DetectorMosquitoMLMejorado()
camera_active = False
detection_active = False
calibration_active = False
calibration_progress = 0

# Estadísticas en tiempo real
current_stats = {
    'detections': 0,
    'total_objects': 0,
    'fps': 0,
    'precision': 'N/A',
    'status': 'Listo',
    'session_time': 0,
    'last_detection_time': None
}

# Control de sesión
session_start = time.time()

def initialize_detector():
    """Inicializa el detector y la cámara"""
    global detector, current_stats
    
    try:
        detector.iniciar_camara(0)
        current_stats['status'] = 'Cámara inicializada ✅'
        print("✅ Detector inicializado correctamente")
        return True
    except Exception as e:
        current_stats['status'] = f'Error de cámara: {str(e)}'
        print(f"❌ Error inicializando detector: {e}")
        return False

def generate_frames():
    """Generador de frames para streaming de video"""
    global camera_active, detection_active, calibration_active, current_stats, calibration_progress
    
    if detector.cap is None:
        if not initialize_detector():
            return
    
    camera_active = True
    frame_count = 0
    start_time = time.time()
    
    print("🎥 Iniciando stream de video...")
    
    try:
        while camera_active:
            ret, frame = detector.cap.read()
            if not ret:
                print("❌ No se pudo leer frame de la cámara")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Calcular FPS cada segundo
            if current_time - start_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                current_stats['fps'] = round(fps, 1)
                frame_count = 0
                start_time = current_time
            
            # Actualizar tiempo de sesión
            current_stats['session_time'] = int(current_time - session_start)
            
            # Procesar frame según el modo activo
            if detection_active:
                frame, detections = process_detection_frame(frame)
                current_stats['detections'] += detections
                if detections > 0:
                    current_stats['last_detection_time'] = datetime.now().strftime('%H:%M:%S')
            
            elif calibration_active:
                frame = process_calibration_frame(frame)
            
            else:
                # Modo normal - solo mostrar video
                add_overlay_info(frame, "Modo visualización")
            
            # Codificar frame como JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Yield frame en formato multipart
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            except Exception as e:
                print(f"Error codificando frame: {e}")
                continue
            
            # Pequeña pausa para controlar FPS
            time.sleep(0.033)  # ~30 FPS máximo
    
    except Exception as e:
        print(f"❌ Error en generate_frames: {e}")
    
    finally:
        camera_active = False
        print("🛑 Stream de video detenido")

def process_detection_frame(frame):
    """Procesa frame para detección de mosquitos"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar sustracción de fondo
    fg_mask = detector.background_subtractor.apply(frame_gray)
    
    # Filtrado morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = 0
    total_objects = 0
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        
        if area < 5:  # Filtrar ruido muy pequeño
            continue
            
        total_objects += 1
        
        # Aplicar filtros del detector mejorado
        if detector.MIN_AREA <= area <= detector.MAX_AREA:
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = w / h if h > 0 else 0
            
            if 1.2 <= aspect_ratio <= 5.0:  # Forma típica de mosquito
                # Calcular características adicionales
                perimeter = cv2.arcLength(contorno, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity < 0.8:  # No muy circular
                        # Es una detección válida
                        detections += 1
                        
                        # Dibujar detección
                        cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                        
                        # Etiqueta con información
                        label = f"🦟 A:{area:.0f} AR:{aspect_ratio:.1f}"
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        # Punto central
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Actualizar estadísticas globales
    current_stats['total_objects'] += total_objects
    
    # Calcular precisión estimada
    if current_stats['total_objects'] > 0:
        precision = current_stats['detections'] / max(current_stats['total_objects'], 1)
        current_stats['precision'] = f"{precision:.1%}"
    
    # Información en el frame
    status_text = f"🔍 Detectando - Encontrados: {detections}"
    add_overlay_info(frame, status_text, detections, total_objects)
    
    current_stats['status'] = f"Detectando... ({detections} mosquitos)"
    
    return frame, detections

def process_calibration_frame(frame):
    """Procesa frame durante calibración"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector.background_subtractor.apply(frame_gray, learningRate=0.01)
    
    # Overlay de calibración
    overlay_height = 80
    overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
    overlay[:] = (0, 100, 200)  # Color de fondo
    
    # Textos de calibración
    cv2.putText(overlay, "CALIBRANDO FONDO", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, "Mantener area sin mosquitos", (20, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Barra de progreso
    if calibration_progress > 0:
        bar_width = int((frame.shape[1] - 40) * (calibration_progress / 100))
        cv2.rectangle(overlay, (20, 60), (20 + bar_width, 70), (0, 255, 0), -1)
        cv2.rectangle(overlay, (20, 60), (frame.shape[1] - 20, 70), (255, 255, 255), 1)
    
    # Combinar overlay con frame
    frame[0:overlay_height] = overlay
    
    current_stats['status'] = f"Calibrando... {calibration_progress:.0f}%"
    
    return frame

def add_overlay_info(frame, status_text, detections=0, total_objects=0):
    """Añade información superpuesta al frame"""
    # Fondo para el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Información principal
    cv2.putText(frame, status_text, (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Información adicional
    fps_text = f"FPS: {current_stats['fps']}"
    cv2.putText(frame, fps_text, (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Timestamp
    timestamp = datetime.now().strftime('%H:%M:%S')
    cv2.putText(frame, timestamp, (frame.shape[1] - 100, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# === RUTAS DE LA APLICACIÓN WEB ===

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream de video en tiempo real"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """API: Iniciar detección"""
    global detection_active, calibration_active
    
    if calibration_active:
        return jsonify({
            'success': False, 
            'message': 'No se puede detectar durante calibración'
        })
    
    if detector.cap is None:
        if not initialize_detector():
            return jsonify({
                'success': False,
                'message': 'Error inicializando cámara'
            })
    
    detection_active = True
    current_stats['status'] = 'Detección iniciada'
    
    return jsonify({
        'success': True, 
        'message': 'Detección iniciada correctamente'
    })

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """API: Detener detección"""
    global detection_active
    
    detection_active = False
    current_stats['status'] = 'Detección detenida'
    
    return jsonify({
        'success': True, 
        'message': 'Detección detenida'
    })

@app.route('/api/calibrate', methods=['POST'])
def calibrate_background():
    """API: Calibrar fondo"""
    global calibration_active, calibration_progress, detection_active
    
    if detection_active:
        return jsonify({
            'success': False, 
            'message': 'Detén la detección primero'
        })
    
    duration = int(request.json.get('duration', 15))
    
    def calibration_thread():
        global calibration_active, calibration_progress
        
        calibration_active = True
        calibration_progress = 0
        
        for i in range(duration):
            time.sleep(1)
            calibration_progress = ((i + 1) / duration) * 100
            current_stats['status'] = f'Calibrando... {calibration_progress:.0f}%'
        
        calibration_active = False
        calibration_progress = 0
        current_stats['status'] = 'Calibración completada ✅'
    
    thread = threading.Thread(target=calibration_thread, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': f'Calibración iniciada por {duration} segundos'
    })

@app.route('/api/capture', methods=['POST'])
def capture_image():
    """API: Capturar imagen actual"""
    if detector.cap is None:
        return jsonify({'success': False, 'message': 'Cámara no disponible'})
    
    try:
        ret, frame = detector.cap.read()
        if not ret:
            return jsonify({'success': False, 'message': 'No se pudo capturar imagen'})
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mosquito_captura_{timestamp}.jpg"
        filepath = os.path.join('capturas', filename)
        
        # Guardar imagen
        cv2.imwrite(filepath, frame)
        
        return jsonify({
            'success': True, 
            'message': f'Imagen capturada: {filename}',
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error capturando: {str(e)}'
        })

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    """API: Resetear estadísticas"""
    global current_stats, session_start
    
    current_stats.update({
        'detections': 0,
        'total_objects': 0,
        'precision': 'N/A',
        'last_detection_time': None
    })
    
    session_start = time.time()
    current_stats['session_time'] = 0
    current_stats['status'] = 'Estadísticas reseteadas'
    
    return jsonify({
        'success': True, 
        'message': 'Estadísticas reseteadas correctamente'
    })

@app.route('/api/stats')
def get_stats():
    """API: Obtener estadísticas actuales"""
    return jsonify(current_stats)

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """API: Configuración del detector"""
    if request.method == 'GET':
        return jsonify({
            'min_area': detector.MIN_AREA,
            'max_area': detector.MAX_AREA,
            'sensitivity': detector.detection_confidence_threshold,
            'movement_threshold': getattr(detector, 'movement_threshold', 5.0)
        })
    
    elif request.method == 'POST':
        try:
            data = request.json
            
            if 'min_area' in data:
                detector.MIN_AREA = max(5, min(100, int(data['min_area'])))
            
            if 'max_area' in data:
                detector.MAX_AREA = max(50, min(2000, int(data['max_area'])))
            
            if 'sensitivity' in data:
                detector.detection_confidence_threshold = max(0.1, min(0.99, float(data['sensitivity'])))
            
            if 'movement_threshold' in data:
                detector.movement_threshold = max(1.0, min(50.0, float(data['movement_threshold'])))
            
            return jsonify({
                'success': True, 
                'message': 'Configuración actualizada correctamente'
            })
        
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Error actualizando configuración: {str(e)}'
            })

@app.route('/capturas/<filename>')
def serve_capture(filename):
    """Servir imágenes capturadas"""
    return send_from_directory('capturas', filename)

@app.route('/api/captures')
def list_captures():
    """API: Listar capturas disponibles"""
    try:
        if not os.path.exists('capturas'):
            return jsonify({'captures': []})
        
        files = [f for f in os.listdir('capturas') 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort(reverse=True)  # Más recientes primero
        
        return jsonify({'captures': files})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# === MANEJO DE ERRORES ===

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

# === FUNCIÓN PRINCIPAL ===

def main():
    """Función principal de la aplicación"""
    print("🌐 INICIANDO APLICACIÓN WEB - DETECTOR DE MOSQUITOS")
    print("=" * 60)
    
    # Verificar archivos necesarios
    if not os.path.exists('detector_mejorado.py'):
        print("❌ ERROR: detector_mejorado.py no encontrado")
        print("   Asegúrate de que esté en la misma carpeta")
        return
    
    # Crear directorios necesarios
    os.makedirs('capturas', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Información de acceso
    print("\n📱 ACCESO A LA APLICACIÓN:")
    print(f"   🏠 Local:      http://localhost:5000")
    print(f"   🌐 Red local:  http://[tu-ip]:5000")
    
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"   Puerto: 5000")
    print(f"   Debug: Desactivado")
    print(f"   Threading: Activado")
    
    print("\n📋 FUNCIONALIDADES:")
    print("   ✅ Stream de video en tiempo real")
    print("   ✅ Detección con ML")
    print("   ✅ Calibración de fondo")
    print("   ✅ Captura de imágenes")
    print("   ✅ API REST completa")
    print("   ✅ Estadísticas en vivo")
    
    print("\n⚠️  IMPORTANTE:")
    print("   - Asegúrate de tener una cámara conectada")
    print("   - Permite acceso de cámara si aparece popup")
    print("   - Presiona Ctrl+C para detener el servidor")
    
    print("\n🚀 Iniciando servidor...")
    
    try:
        app.run(
            host='0.0.0.0',  # Accesible desde la red local
            port=5000,
            debug=False,     # Desactivar debug en producción
            threaded=True,   # Permitir múltiples conexiones
            use_reloader=False  # Evitar problemas con threading
        )
    
    except KeyboardInterrupt:
        print("\n\n🛑 SERVIDOR DETENIDO")
        print("👋 ¡Gracias por usar el Detector de Mosquitos ML!")
    
    except Exception as e:
        print(f"\n❌ ERROR INICIANDO SERVIDOR: {e}")
    
    finally:
        # Limpiar recursos
        global camera_active
        camera_active = False
        
        if detector.cap:
            detector.cap.release()
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("🧹 Recursos liberados correctamente")

if __name__ == '__main__':
    main()