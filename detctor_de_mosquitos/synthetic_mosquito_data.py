import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.datasets import make_classification
from scipy.stats import norm, gamma, beta
import cv2

class GeneradorDatosSinteticos:
    def __init__(self):
        # Parámetros basados en características reales de mosquitos
        self.mosquito_params = {
            'area': {'min': 15, 'max': 150, 'mean': 45, 'std': 20},
            'aspect_ratio': {'min': 1.5, 'max': 4.0, 'mean': 2.2, 'std': 0.6},
            'solidity': {'min': 0.4, 'max': 0.9, 'mean': 0.65, 'std': 0.15},
            'velocidad': {'min': 20, 'max': 200, 'mean': 80, 'std': 30},
            'cambios_direccion': {'min': 2, 'max': 8, 'mean': 4, 'std': 1.5},
            'elongacion': {'min': 1.8, 'max': 5.0, 'mean': 2.5, 'std': 0.8},
            'mean_intensity': {'min': 60, 'max': 180, 'mean': 110, 'std': 25}
        }
        
        self.no_mosquito_params = {
            'area': {'min': 5, 'max': 500, 'mean': 120, 'std': 80},
            'aspect_ratio': {'min': 0.8, 'max': 2.5, 'mean': 1.4, 'std': 0.4},
            'solidity': {'min': 0.6, 'max': 1.0, 'mean': 0.85, 'std': 0.1},
            'velocidad': {'min': 0, 'max': 50, 'mean': 15, 'std': 12},
            'cambios_direccion': {'min': 0, 'max': 3, 'mean': 1, 'std': 0.8},
            'elongacion': {'min': 1.0, 'max': 2.2, 'mean': 1.3, 'std': 0.3},
            'mean_intensity': {'min': 40, 'max': 200, 'mean': 140, 'std': 35}
        }
    
    def generar_caracteristica(self, params, distribucion='normal'):
        """Genera una característica usando diferentes distribuciones"""
        if distribucion == 'normal':
            valor = np.random.normal(params['mean'], params['std'])
        elif distribucion == 'gamma':
            # Para características que no pueden ser negativas
            shape = (params['mean'] / params['std']) ** 2
            scale = (params['std'] ** 2) / params['mean']
            valor = np.random.gamma(shape, scale)
        elif distribucion == 'beta':
            # Para características normalizadas entre 0 y 1
            valor = np.random.beta(2, 2) * (params['max'] - params['min']) + params['min']
        else:
            valor = np.random.uniform(params['min'], params['max'])
        
        # Asegurar que esté dentro de los límites
        return np.clip(valor, params['min'], params['max'])
    
    def generar_muestra_mosquito(self):
        """Genera una muestra sintética de mosquito"""
        caracteristicas = []
        
        # Características geométricas básicas
        area = self.generar_caracteristica(self.mosquito_params['area'], 'gamma')
        perimetro = 2 * np.sqrt(np.pi * area) * np.random.uniform(1.2, 1.8)  # Factor de forma
        area_perimetro_ratio = area / (perimetro ** 2)
        
        caracteristicas.extend([area, perimetro, area_perimetro_ratio])
        
        # Dimensiones del rectángulo
        aspect_ratio = self.generar_caracteristica(self.mosquito_params['aspect_ratio'])
        height = np.sqrt(area / aspect_ratio)
        width = area / height
        extent = area / (width * height) * np.random.uniform(0.6, 0.9)  # Los mosquitos no llenan todo el rectángulo
        
        caracteristicas.extend([width, height, aspect_ratio, extent])
        
        # Características de forma
        solidity = self.generar_caracteristica(self.mosquito_params['solidity'])
        compactness = area_perimetro_ratio * np.random.uniform(0.8, 1.2)  # Relacionado con área/perímetro
        
        caracteristicas.extend([solidity, compactness])
        
        # Momentos de Hu (simulados con distribuciones apropiadas)
        hu_moments = []
        for i in range(7):
            if i < 2:
                # Primeros momentos más estables
                hu = np.random.normal(-2, 1)
            else:
                # Momentos de orden superior más variables
                hu = np.random.normal(-8, 3)
            hu_moments.append(hu)
        
        caracteristicas.extend(hu_moments)
        
        # Características de intensidad
        mean_intensity = self.generar_caracteristica(self.mosquito_params['mean_intensity'])
        std_intensity = mean_intensity * np.random.uniform(0.15, 0.35)  # Variabilidad proporcional
        
        caracteristicas.extend([mean_intensity, std_intensity])
        
        # Características de movimiento (los mosquitos se mueven erráticamente)
        velocidad = self.generar_caracteristica(self.mosquito_params['velocidad'], 'gamma')
        aceleracion = velocidad * np.random.uniform(0.1, 0.4)  # Aceleración proporcional a velocidad
        cambios_direccion = self.generar_caracteristica(self.mosquito_params['cambios_direccion'], 'gamma')
        
        caracteristicas.extend([velocidad, aceleracion, cambios_direccion])
        
        # Características adicionales
        elongacion = self.generar_caracteristica(self.mosquito_params['elongacion'])
        densidad = solidity * np.random.uniform(0.7, 1.0)  # Relacionado con solidez
        
        caracteristicas.extend([elongacion, densidad])
        
        return np.array(caracteristicas)
    
    def generar_muestra_no_mosquito(self):
        """Genera una muestra sintética de no-mosquito (polvo, otros insectos, ruido)"""
        caracteristicas = []
        
        # Características más variables y diferentes a mosquitos
        area = self.generar_caracteristica(self.no_mosquito_params['area'], 'gamma')
        perimetro = 2 * np.sqrt(np.pi * area) * np.random.uniform(0.8, 2.5)  # Más variable
        area_perimetro_ratio = area / (perimetro ** 2) if perimetro > 0 else 0
        
        caracteristicas.extend([area, perimetro, area_perimetro_ratio])
        
        # Dimensiones más compactas típicamente
        aspect_ratio = self.generar_caracteristica(self.no_mosquito_params['aspect_ratio'])
        height = np.sqrt(area / aspect_ratio)
        width = area / height if height > 0 else 1
        extent = area / (width * height) * np.random.uniform(0.4, 1.0) if (width * height) > 0 else 0
        
        caracteristicas.extend([width, height, aspect_ratio, extent])
        
        # Forma más compacta
        solidity = self.generar_caracteristica(self.no_mosquito_params['solidity'])
        compactness = area_perimetro_ratio * np.random.uniform(0.5, 1.5)
        
        caracteristicas.extend([solidity, compactness])
        
        # Momentos de Hu diferentes
        hu_moments = []
        for i in range(7):
            if i < 2:
                hu = np.random.normal(-1.5, 1.5)
            else:
                hu = np.random.normal(-6, 4)
            hu_moments.append(hu)
        
        caracteristicas.extend(hu_moments)
        
        # Intensidades diferentes
        mean_intensity = self.generar_caracteristica(self.no_mosquito_params['mean_intensity'])
        std_intensity = mean_intensity * np.random.uniform(0.1, 0.5)
        
        caracteristicas.extend([mean_intensity, std_intensity])
        
        # Movimiento más estable (polvo, objetos estáticos)
        velocidad = self.generar_caracteristica(self.no_mosquito_params['velocidad'], 'gamma')
        aceleracion = velocidad * np.random.uniform(0.0, 0.2)
        cambios_direccion = self.generar_caracteristica(self.no_mosquito_params['cambios_direccion'])
        
        caracteristicas.extend([velocidad, aceleracion, cambios_direccion])
        
        # Menos elongados
        elongacion = self.generar_caracteristica(self.no_mosquito_params['elongacion'])
        densidad = solidity * np.random.uniform(0.8, 1.0)
        
        caracteristicas.extend([elongacion, densidad])
        
        return np.array(caracteristicas)
    
    def generar_dataset(self, n_mosquitos=500, n_no_mosquitos=500, ruido_factor=0.1):
        """Genera un dataset sintético completo"""
        print(f"🎯 Generando dataset sintético...")
        print(f"  Mosquitos: {n_mosquitos}")
        print(f"  No-mosquitos: {n_no_mosquitos}")
        print(f"  Factor de ruido: {ruido_factor}")
        
        datos = []
        etiquetas = []
        
        # Generar muestras de mosquitos
        print("🦟 Generando muestras de mosquitos...")
        for i in range(n_mosquitos):
            muestra = self.generar_muestra_mosquito()
            
            # Añadir ruido
            if ruido_factor > 0:
                ruido = np.random.normal(0, ruido_factor, len(muestra))
                muestra = muestra + ruido
            
            datos.append(muestra)
            etiquetas.append(1)
            
            if (i + 1) % 100 == 0:
                print(f"  Generados: {i + 1}/{n_mosquitos}")
        
        # Generar muestras de no-mosquitos
        print("❌ Generando muestras de no-mosquitos...")
        for i in range(n_no_mosquitos):
            muestra = self.generar_muestra_no_mosquito()
            
            # Añadir ruido
            if ruido_factor > 0:
                ruido = np.random.normal(0, ruido_factor, len(muestra))
                muestra = muestra + ruido
            
            datos.append(muestra)
            etiquetas.append(0)
            
            if (i + 1) % 100 == 0:
                print(f"  Generados: {i + 1}/{n_no_mosquitos}")
        
        # Mezclar los datos
        indices = np.random.permutation(len(datos))
        datos = np.array(datos)[indices]
        etiquetas = np.array(etiquetas)[indices]
        
        return datos, etiquetas
    
    def visualizar_dataset(self, datos, etiquetas, guardar=True):
        """Visualiza las características del dataset generado"""
        nombres_caracteristicas = [
            'area', 'perimetro', 'area_perimetro_ratio', 'width', 'height', 
            'aspect_ratio', 'extent', 'solidity', 'compactness',
            'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
            'mean_intensity', 'std_intensity', 'velocidad', 'aceleracion', 
            'cambios_direccion', 'elongacion', 'densidad'
        ]
        
        # Seleccionar características clave para visualizar
        caracteristicas_clave = [0, 5, 7, 19, 21, 22, 17]  # area, aspect_ratio, solidity, velocidad, cambios_direccion, elongacion, mean_intensity
        nombres_clave = [nombres_caracteristicas[i] for i in caracteristicas_clave]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Dataset Sintético - Distribución de Características', fontsize=16)
        
        for i, idx in enumerate(caracteristicas_clave + [18]):  # Añadir std_intensity
            if i >= 8:
                break
            
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Separar datos por clase
            mosquitos = datos[etiquetas == 1, idx] if idx < datos.shape[1] else []
            no_mosquitos = datos[etiquetas == 0, idx] if idx < datos.shape[1] else []
            
            if len(mosquitos) > 0 and len(no_mosquitos) > 0:
                ax.hist(no_mosquitos, alpha=0.7, label='No-Mosquitos', color='blue', bins=30)
                ax.hist(mosquitos, alpha=0.7, label='Mosquitos', color='red', bins=30)
                
                nombre = nombres_caracteristicas[idx] if idx < len(nombres_caracteristicas) else f'Feature_{idx}'
                ax.set_title(nombre)
                ax.set_xlabel('Valor')
                ax.set_ylabel('Frecuencia')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Eliminar subplot vacío
        if len(caracteristicas_clave) < 8:
            axes[2, 2].remove()
        
        plt.tight_layout()
        
        if guardar:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_sintetico_viz_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Visualización guardada: {filename}")
        
        plt.show()
    
    def guardar_dataset(self, datos, etiquetas, prefijo="sintetico"):
        """Guarda el dataset en formato compatible con el detector ML"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"datos_mosquitos_{prefijo}_{timestamp}.pkl"
        
        datos_dict = {
            'caracteristicas': datos,
            'etiquetas': etiquetas,
            'timestamp': time.time(),
            'tipo': 'sintetico',
            'info': {
                'total_muestras': len(datos),
                'mosquitos': int(np.sum(etiquetas)),
                'no_mosquitos': int(len(etiquetas) - np.sum(etiquetas)),
                'num_caracteristicas': datos.shape[1]
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(datos_dict, f)
        
        print(f"💾 Dataset guardado: {filename}")
        print(f"📊 Estadísticas:")
        print(f"  Total: {datos_dict['info']['total_muestras']} muestras")
        print(f"  Mosquitos: {datos_dict['info']['mosquitos']}")
        print(f"  No-mosquitos: {datos_dict['info']['no_mosquitos']}")
        print(f"  Características: {datos_dict['info']['num_caracteristicas']}")
        
        return filename
    
    def generar_dataset_balanceado(self, total_muestras=1000, balance=0.3):
        """Genera un dataset con balance controlado"""
        n_mosquitos = int(total_muestras * balance)
        n_no_mosquitos = total_muestras - n_mosquitos
        
        return self.generar_dataset(n_mosquitos, n_no_mosquitos)
    
    def generar_dataset_desbalanceado(self, total_muestras=1000, proporcion_mosquitos=0.1):
        """Genera un dataset desbalanceado (más realista)"""
        n_mosquitos = int(total_muestras * proporcion_mosquitos)
        n_no_mosquitos = total_muestras - n_mosquitos
        
        return self.generar_dataset(n_mosquitos, n_no_mosquitos)
    
    def crear_variaciones_temporales(self, n_sets=5, muestras_por_set=200):
        """Crea múltiples sets con variaciones para simular datos temporales"""
        datasets = []
        
        for i in range(n_sets):
            print(f"\n📅 Generando set temporal {i+1}/{n_sets}")
            
            # Variar parámetros ligeramente para cada set
            factor_variacion = 1 + np.random.normal(0, 0.1)
            
            # Modificar parámetros temporalmente
            params_temp = {}
            for categoria, params in self.mosquito_params.items():
                params_temp[categoria] = {
                    k: v * factor_variacion for k, v in params.items()
                }
            
            # Guardar parámetros originales
            params_originales = self.mosquito_params.copy()
            self.mosquito_params = params_temp
            
            # Generar dataset
            datos, etiquetas = self.generar_dataset_balanceado(muestras_por_set)
            datasets.append((datos, etiquetas))
            
            # Restaurar parámetros
            self.mosquito_params = params_originales
        
        return datasets

def main():
    generador = GeneradorDatosSinteticos()
    
    print("🧪 GENERADOR DE DATOS SINTÉTICOS - MOSQUITOS")
    print("=" * 50)
    
    while True:
        print("\n📋 OPCIONES:")
        print("1. Generar dataset balanceado")
        print("2. Generar dataset desbalanceado (realista)")
        print("3. Generar dataset personalizado")
        print("4. Crear variaciones temporales")
        print("5. Generar dataset de prueba rápido")
        print("6. Salir")
        
        opcion = input("\nSelecciona una opción (1-6): ").strip()
        
        if opcion == "1":
            total = input("Total de muestras (default: 1000): ").strip()
            total = int(total) if total.isdigit() else 1000
            
            balance = input("Proporción de mosquitos (default: 0.3): ").strip()
            try:
                balance = float(balance)
            except:
                balance = 0.3
            
            print(f"\n🎯 Generando dataset balanceado...")
            datos, etiquetas = generador.generar_dataset_balanceado(total, balance)
            
            # Visualizar
            generador.visualizar_dataset(datos, etiquetas)
            
            # Guardar
            filename = generador.guardar_dataset(datos, etiquetas, "balanceado")
            print(f"✅ Dataset generado: {filename}")
        
        elif opcion == "2":
            total = input("Total de muestras (default: 1000): ").strip()
            total = int(total) if total.isdigit() else 1000
            
            proporcion = input("Proporción de mosquitos (default: 0.1): ").strip()
            try:
                proporcion = float(proporcion)
            except:
                proporcion = 0.1
            
            print(f"\n🎯 Generando dataset desbalanceado...")
            datos, etiquetas = generador.generar_dataset_desbalanceado(total, proporcion)
            
            # Visualizar
            generador.visualizar_dataset(datos, etiquetas)
            
            # Guardar
            filename = generador.guardar_dataset(datos, etiquetas, "desbalanceado")
            print(f"✅ Dataset generado: {filename}")
        
        elif opcion == "3":
            n_mosquitos = input("Número de mosquitos: ").strip()
            n_mosquitos = int(n_mosquitos) if n_mosquitos.isdigit() else 300
            
            n_no_mosquitos = input("Número de no-mosquitos: ").strip()
            n_no_mosquitos = int(n_no_mosquitos) if n_no_mosquitos.isdigit() else 700
            
            ruido = input("Factor de ruido (0.0-0.5, default: 0.1): ").strip()
            try:
                ruido = float(ruido)
            except:
                ruido = 0.1
            
            print(f"\n🎯 Generando dataset personalizado...")
            datos, etiquetas = generador.generar_dataset(n_mosquitos, n_no_mosquitos, ruido)
            
            # Visualizar
            generador.visualizar_dataset(datos, etiquetas)
            
            # Guardar
            filename = generador.guardar_dataset(datos, etiquetas, "personalizado")
            print(f"✅ Dataset generado: {filename}")
        
        elif opcion == "4":
            n_sets = input("Número de sets temporales (default: 5): ").strip()
            n_sets = int(n_sets) if n_sets.isdigit() else 5
            
            muestras = input("Muestras por set (default: 200): ").strip()
            muestras = int(muestras) if muestras.isdigit() else 200
            
            print(f"\n📅 Generando variaciones temporales...")
            datasets = generador.crear_variaciones_temporales(n_sets, muestras)
            
            # Combinar todos los sets
            todos_datos = np.vstack([datos for datos, _ in datasets])
            todas_etiquetas = np.concatenate([etiquetas for _, etiquetas in datasets])
            
            # Visualizar conjunto combinado
            generador.visualizar_dataset(todos_datos, todas_etiquetas)
            
            # Guardar
            filename = generador.guardar_dataset(todos_datos, todas_etiquetas, "temporal")
            print(f"✅ Dataset temporal generado: {filename}")
        
        elif opcion == "5":
            print(f"\n⚡ Generando dataset de prueba rápido...")
            datos, etiquetas = generador.generar_dataset(100, 200, 0.05)
            
            # Guardar sin visualizar
            filename = generador.guardar_dataset(datos, etiquetas, "prueba")
            print(f"✅ Dataset de prueba generado: {filename}")
        
        elif opcion == "6":
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()