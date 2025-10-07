#!/usr/bin/env python3
"""
Script de instalacion y configuracion automatica para el Detector ML de Mosquitos
Version corregida para Windows con manejo de UTF-8
"""

import subprocess
import sys
import os
import platform
import time
from pathlib import Path

class ConfiguradorDetectorML:
    def __init__(self):
        self.sistema = platform.system()
        self.python_version = sys.version_info
        self.dependencias_requeridas = [
            'numpy>=1.21.0',
            'opencv-python>=4.5.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'pandas>=1.3.0',
            'scipy>=1.7.0',
            'pyaudio>=0.2.11'
        ]
        self.dependencias_opcionales = [
            'tensorflow>=2.8.0',
            'torch>=1.10.0',
            'jupyter>=1.0.0',
            'plotly>=5.0.0'
        ]
    
    def verificar_python(self):
        """Verifica que la version de Python sea compatible"""
        print("Verificando version de Python...")
        
        if self.python_version < (3, 7):
            print(f"Error: Se requiere Python 3.7 o superior. Version actual: {sys.version}")
            return False
        
        print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def instalar_dependencias(self, lista_dependencias, opcional=False):
        """Instala las dependencias requeridas"""
        tipo = "opcionales" if opcional else "requeridas"
        print(f"\nInstalando dependencias {tipo}...")
        
        for dependencia in lista_dependencias:
            try:
                print(f"  Instalando: {dependencia}")
                resultado = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dependencia
                ], capture_output=True, text=True, timeout=300)
                
                if resultado.returncode == 0:
                    print(f"  OK: {dependencia}")
                else:
                    if opcional:
                        print(f"  OMITIDA: {dependencia} (opcional)")
                    else:
                        print(f"  ERROR instalando {dependencia}")
                        print(f"     {resultado.stderr}")
                        return False
                        
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT instalando {dependencia}")
                if not opcional:
                    return False
            except Exception as e:
                print(f"  ERROR: {e}")
                if not opcional:
                    return False
        
        return True
    
    def configurar_pyaudio(self):
        """Configuracion especial para PyAudio segun el sistema operativo"""
        print("\nConfigurando PyAudio...")
        
        if self.sistema == "Windows":
            print("  Sistema Windows detectado")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyaudio'], 
                             check=True, capture_output=True)
                print("  PyAudio instalado correctamente")
                return True
            except:
                print("  Si PyAudio falla, instala manualmente desde:")
                print("     https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
                return False
                
        elif self.sistema == "Darwin":  # macOS
            print("  Sistema macOS detectado")
            print("  Instalando dependencias del sistema...")
            try:
                subprocess.run(['brew', '--version'], check=True, capture_output=True)
                subprocess.run(['brew', 'install', 'portaudio'], check=True, capture_output=True)
            except:
                print("  Instala Homebrew y ejecuta: brew install portaudio")
            
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyaudio'], 
                             check=True, capture_output=True)
                print("  PyAudio instalado correctamente")
                return True
            except:
                print("  Error instalando PyAudio")
                return False
                
        elif self.sistema == "Linux":
            print("  Sistema Linux detectado")
            print("  Instalando dependencias del sistema...")
            try:
                gestores = [
                    (['apt-get', 'update'], ['apt-get', 'install', '-y', 'portaudio19-dev']),
                    (['yum', 'install', '-y', 'portaudio-devel']),
                    (['pacman', '-S', 'portaudio'])
                ]
                
                for comandos in gestores:
                    try:
                        for cmd in comandos:
                            subprocess.run(cmd, check=True, capture_output=True)
                        break
                    except:
                        continue
                
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyaudio'], 
                             check=True, capture_output=True)
                print("  PyAudio instalado correctamente")
                return True
            except:
                print("  Instala manualmente: sudo apt-get install portaudio19-dev")
                return False
        
        return True
    
    def verificar_camara(self):
        """Verifica que la camara este disponible"""
        print("\nVerificando camara...")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("  Camara funcionando correctamente")
                    cap.release()
                    return True
                else:
                    print("  Camara detectada pero no puede leer frames")
            else:
                print("  No se puede acceder a la camara")
            
            cap.release()
            return False
            
        except Exception as e:
            print(f"  Error verificando camara: {e}")
            return False
    
    def verificar_microfono(self):
        """Verifica que el microfono este disponible"""
        print("\nVerificando microfono...")
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            
            dispositivos_entrada = []
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    dispositivos_entrada.append(info['name'])
            
            p.terminate()
            
            if dispositivos_entrada:
                print(f"  {len(dispositivos_entrada)} dispositivos de audio encontrados")
                for i, dispositivo in enumerate(dispositivos_entrada[:3]):
                    print(f"    {i+1}. {dispositivo}")
                return True
            else:
                print("  No se encontraron dispositivos de entrada de audio")
                return False
                
        except Exception as e:
            print(f"  Error verificando microfono: {e}")
            return False
    
    def crear_estructura_directorios(self):
        """Crea la estructura de directorios necesaria"""
        print("\nCreando estructura de directorios...")
        
        directorios = [
            'datos',
            'modelos',
            'resultados',
            'reportes',
            'capturas'
        ]
        
        for directorio in directorios:
            Path(directorio).mkdir(exist_ok=True)
            print(f"  Creado: {directorio}/")
        
        return True
    
    def crear_archivo_config(self):
        """Crea archivo de configuracion inicial"""
        print("\nCreando archivo de configuracion...")
        
        config = f"""# Configuracion del Detector ML de Mosquitos
# Generado automaticamente el {time.strftime('%Y-%m-%d %H:%M:%S')}

[GENERAL]
version = 1.0
sistema = {self.sistema}
python_version = {self.python_version.major}.{self.python_version.minor}

[CAMARA]
device_id = 0
width = 640
height = 480
fps = 30

[AUDIO]
sample_rate = 44100
chunk_size = 4096
channels = 1

[DETECCION]
min_area = 5
max_area = 500
confidence_threshold = 0.7

[ML]
default_model = RandomForest
test_size = 0.2
cv_folds = 5

[PATHS]
datos_dir = datos
modelos_dir = modelos
resultados_dir = resultados
reportes_dir = reportes
capturas_dir = capturas
"""
        
        with open('config.ini', 'w', encoding='utf-8') as f:
            f.write(config)
        
        print("  config.ini creado correctamente")
        return True
    
    def crear_script_inicio(self):
        """Crea script de inicio rapido"""
        print("\nCreando script de inicio...")
        
        # Script sin emojis para evitar problemas de codificacion
        script = """#!/usr/bin/env python3
# Script de inicio rapido para el Detector ML de Mosquitos

import sys
import os

def main():
    print("DETECTOR ML DE MOSQUITOS - INICIO RAPIDO")
    print("=" * 45)
    
    opciones = {
        "1": ("Detector ML completo", "python mosquito_ml_detector.py"),
        "2": ("Generar datos sinteticos", "python synthetic_mosquito_data.py"),
        "3": ("Analisis de datos", "python mosquito_analysis_tools.py"),
        "4": ("Configuracion", "python setup_mosquito_detector.py")
    }
    
    print("\\nOpciones disponibles:")
    for key, (desc, _) in opciones.items():
        print(f"  {key}. {desc}")
    
    seleccion = input("\\nSelecciona una opcion (1-4): ").strip()
    
    if seleccion in opciones:
        _, comando = opciones[seleccion]
        print(f"\\nEjecutando: {comando}")
        os.system(comando)
    else:
        print("Opcion invalida")

if __name__ == "__main__":
    main()
"""
        
        with open('inicio_rapido.py', 'w', encoding='utf-8') as f:
            f.write(script)
        
        # Hacer ejecutable en sistemas Unix
        if self.sistema in ['Linux', 'Darwin']:
            os.chmod('inicio_rapido.py', 0o755)
        
        print("  inicio_rapido.py creado correctamente")
        return True
    
    def crear_requirements(self):
        """Crea archivo requirements.txt"""
        print("\nCreando requirements.txt...")
        
        requirements = "\n".join(self.dependencias_requeridas + [
            "# Dependencias opcionales",
            "# " + "\n# ".join(self.dependencias_opcionales)
        ])
        
        with open('requirements.txt', 'w', encoding='utf-8') as f:
            f.write(requirements)
        
        print("  requirements.txt creado correctamente")
        return True
    
    def crear_readme(self):
        """Crea archivo README con instrucciones"""
        print("\nCreando README.md...")
        
        readme = f"""# Detector ML de Mosquitos

Sistema de deteccion automatica de mosquitos usando Machine Learning y vision computacional.

## Inicio Rapido

```bash
python inicio_rapido.py
```

## Componentes

### 1. Detector Principal (`mosquito_ml_detector.py`)
- Recoleccion de datos de entrenamiento
- Entrenamiento de modelos ML
- Deteccion en tiempo real

### 2. Generador de Datos (`synthetic_mosquito_data.py`)
- Genera datos sinteticos para entrenamiento inicial
- Util para pruebas y desarrollo

### 3. Herramientas de Analisis (`mosquito_analysis_tools.py`)
- Analisis estadistico de datos
- Visualizaciones y reportes
- Evaluacion de modelos

## Instalacion

### Automatica
```bash
python setup_mosquito_detector.py
```

### Manual
```bash
pip install -r requirements.txt
```

## Uso

### 1. Generar datos iniciales
```bash
python synthetic_mosquito_data.py
```

### 2. Entrenar modelo
```bash
python mosquito_ml_detector.py
# Seleccionar opcion 2 (Entrenar nuevo modelo)
```

### 3. Deteccion en tiempo real
```bash
python mosquito_ml_detector.py
# Seleccionar opcion 4 (Deteccion en tiempo real)
```

## Configuracion

Edita `config.ini` para ajustar parametros:
- Configuracion de camara
- Parametros de deteccion
- Rutas de archivos

## Caracteristicas

- **Deteccion por ML**: Utiliza caracteristicas geometricas y de movimiento
- **Multiples algoritmos**: Random Forest, SVM, Gradient Boosting
- **Analisis avanzado**: PCA, correlaciones, importancia de caracteristicas
- **Datos sinteticos**: Generacion automatica para entrenamiento inicial
- **Tiempo real**: Deteccion y tracking en vivo

## Estructura de Archivos

```
├── mosquito_ml_detector.py      # Detector principal
├── synthetic_mosquito_data.py   # Generador de datos
├── mosquito_analysis_tools.py   # Herramientas de analisis
├── setup_mosquito_detector.py   # Configuracion automatica
├── inicio_rapido.py            # Script de inicio
├── config.ini                  # Configuracion
├── requirements.txt            # Dependencias
├── datos/                      # Datos de entrenamiento
├── modelos/                    # Modelos entrenados
├── resultados/                 # Resultados de deteccion
└── reportes/                   # Reportes de analisis
```

Sistema configurado el {time.strftime('%Y-%m-%d')} para {self.sistema}.
"""
        
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme)
        
        print("  README.md creado correctamente")
        return True
    
    def ejecutar_pruebas(self):
        """Ejecuta pruebas basicas del sistema"""
        print("\nEjecutando pruebas del sistema...")
        
        pruebas = [
            ("Importacion de OpenCV", self._test_opencv),
            ("Importacion de scikit-learn", self._test_sklearn),
            ("Importacion de matplotlib", self._test_matplotlib),
            ("Funciones de NumPy", self._test_numpy)
        ]
        
        resultados = []
        for nombre, prueba in pruebas:
            try:
                resultado = prueba()
                print(f"  OK: {nombre}")
                resultados.append(True)
            except Exception as e:
                print(f"  ERROR: {nombre}: {e}")
                resultados.append(False)
        
        exito = sum(resultados)
        total = len(resultados)
        print(f"\nPruebas completadas: {exito}/{total} exitosas")
        
        return exito == total
    
    def _test_opencv(self):
        import cv2
        return True
    
    def _test_sklearn(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10, n_features=4, random_state=42)
        clf = RandomForestClassifier(n_estimators=2, random_state=42)
        clf.fit(X, y)
        return True
    
    def _test_matplotlib(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return True
    
    def _test_numpy(self):
        import numpy as np
        arr = np.random.random((10, 5))
        return arr.shape == (10, 5)
    
    def mostrar_resumen(self, resultados):
        """Muestra resumen final de la instalacion"""
        print("\n" + "="*60)
        print("RESUMEN DE INSTALACION")
        print("="*60)
        
        for componente, exito in resultados.items():
            estado = "OK" if exito else "ERROR"
            print(f"  {estado}: {componente}")
        
        if all(resultados.values()):
            print("\nINSTALACION COMPLETADA EXITOSAMENTE!")
            print("\nPara empezar:")
            print("  1. python inicio_rapido.py")
            print("  2. Selecciona 'Generar datos sinteticos' para crear datos iniciales")
            print("  3. Luego 'Detector ML completo' para entrenar y detectar")
        else:
            print("\nInstalacion completada con algunos problemas")
            print("   Revisa los errores anteriores y ejecuta nuevamente si es necesario")
        
        print("\nDocumentacion: README.md")
        print("Configuracion: config.ini")

def main():
    print("CONFIGURADOR AUTOMATICO - DETECTOR ML MOSQUITOS")
    print("="*55)
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()
    
    configurador = ConfiguradorDetectorML()
    resultados = {}
    
    # Verificar Python
    if not configurador.verificar_python():
        print("Saliendo debido a version incompatible de Python")
        return
    
    # Menu de opciones
    print("OPCIONES DE INSTALACION:")
    print("1. Instalacion completa (recomendada)")
    print("2. Solo dependencias basicas")
    print("3. Solo configuracion de archivos")
    print("4. Solo verificacion del sistema")
    
    opcion = input("\nSelecciona una opcion (1-4): ").strip()
    
    if opcion == "1":
        # Instalacion completa
        resultados['Dependencias basicas'] = configurador.instalar_dependencias(
            configurador.dependencias_requeridas)
        resultados['PyAudio'] = configurador.configurar_pyaudio()
        resultados['Dependencias opcionales'] = configurador.instalar_dependencias(
            configurador.dependencias_opcionales, opcional=True)
        resultados['Verificacion camara'] = configurador.verificar_camara()
        resultados['Verificacion microfono'] = configurador.verificar_microfono()
        resultados['Estructura directorios'] = configurador.crear_estructura_directorios()
        resultados['Archivo configuracion'] = configurador.crear_archivo_config()
        resultados['Script inicio'] = configurador.crear_script_inicio()
        resultados['Requirements'] = configurador.crear_requirements()
        resultados['README'] = configurador.crear_readme()
        resultados['Pruebas sistema'] = configurador.ejecutar_pruebas()
    
    elif opcion == "2":
        # Solo dependencias basicas
        resultados['Dependencias basicas'] = configurador.instalar_dependencias(
            configurador.dependencias_requeridas)
        resultados['PyAudio'] = configurador.configurar_pyaudio()
    
    elif opcion == "3":
        # Solo archivos de configuracion
        resultados['Estructura directorios'] = configurador.crear_estructura_directorios()
        resultados['Archivo configuracion'] = configurador.crear_archivo_config()
        resultados['Script inicio'] = configurador.crear_script_inicio()
        resultados['Requirements'] = configurador.crear_requirements()
        resultados['README'] = configurador.crear_readme()
    
    elif opcion == "4":
        # Solo verificacion
        resultados['Verificacion camara'] = configurador.verificar_camara()
        resultados['Verificacion microfono'] = configurador.verificar_microfono()
        resultados['Pruebas sistema'] = configurador.ejecutar_pruebas()
    
    else:
        print("Opcion invalida")
        return
    
    # Mostrar resumen
    configurador.mostrar_resumen(resultados)

if __name__ == "__main__":
    main()