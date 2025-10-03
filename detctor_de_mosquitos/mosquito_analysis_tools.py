import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import cv2
import os
from datetime import datetime, timedelta
import json

class AnalizadorDatosMosquitos:
    def __init__(self):
        self.datos = None
        self.modelo_datos = None
        self.nombres_caracteristicas = [
            'area', 'perimetro', 'area_perimetro_ratio', 'width', 'height', 
            'aspect_ratio', 'extent', 'solidity', 'compactness',
            'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
            'mean_intensity', 'std_intensity', 'velocidad', 'aceleracion', 
            'cambios_direccion', 'elongacion', 'densidad'
        ]
    
    def cargar_datos(self, archivo_datos):
        """Carga datos de entrenamiento para análisis"""
        with open(archivo_datos, 'rb') as f:
            self.datos = pickle.load(f)
        print(f"✅ Datos cargados: {len(self.datos['caracteristicas'])} muestras")
        
    def cargar_modelo(self, archivo_modelo):
        """Carga modelo para análisis"""
        with open(archivo_modelo, 'rb') as f:
            self.modelo_datos = pickle.load(f)
        print(f"✅ Modelo cargado")
    
    def estadisticas_basicas(self):
        """Muestra estadísticas básicas de los datos"""
        if self.datos is None:
            print("❌ No hay datos cargados")
            return
        
        X = self.datos['caracteristicas']
        y = self.datos['etiquetas']
        
        print("\n📊 ESTADÍSTICAS BÁSICAS:")
        print(f"  Total de muestras: {len(X)}")
        print(f"  Mosquitos: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"  No-mosquitos: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        print(f"  Número de características: {X.shape[1]}")
        
        # Crear DataFrame para análisis más fácil
        df = pd.DataFrame(X[:, :len(self.nombres_caracteristicas)], 
                         columns=self.nombres_caracteristicas)
        df['es_mosquito'] = y
        
        print("\n📈 ESTADÍSTICAS POR CARACTERÍSTICA:")
        print("=" * 60)
        
        for col in self.nombres_caracteristicas:
            if col in df.columns:
                mosquito_vals = df[df['es_mosquito'] == 1][col]
                no_mosquito_vals = df[df['es_mosquito'] == 0][col]
                
                print(f"\n{col}:")
                if len(mosquito_vals) > 0:
                    print(f"  Mosquitos    - Media: {mosquito_vals.mean():.3f}, Std: {mosquito_vals.std():.3f}")
                if len(no_mosquito_vals) > 0:
                    print(f"  No-mosquitos - Media: {no_mosquito_vals.mean():.3f}, Std: {no_mosquito_vals.std():.3f}")
        
        return df
    
    def visualizar_distribucion_caracteristicas(self, guardar=True):
        """Visualiza la distribución de características"""
        if self.datos is None:
            print("❌ No hay datos cargados")
            return
        
        df = self.estadisticas_basicas()
        
        # Seleccionar características más importantes para visualizar
        caracteristicas_clave = ['area', 'aspect_ratio', 'solidity', 'velocidad', 
                               'cambios_direccion', 'elongacion', 'mean_intensity']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Distribución de Características - Mosquitos vs No-Mosquitos', fontsize=16)
        
        for i, caracteristica in enumerate(caracteristicas_clave + ['compactness', 'densidad'][:2]):
            if i >= 9:  # Solo 9 subplots
                break
                
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if caracteristica in df.columns:
                # Histogramas superpuestos
                mosquitos = df[df['es_mosquito'] == 1][caracteristica]
                no_mosquitos = df[df['es_mosquito'] == 0][caracteristica]
                
                ax.hist(no_mosquitos, alpha=0.7, label='No-Mosquitos', color='blue', bins=20)
                ax.hist(mosquitos, alpha=0.7, label='Mosquitos', color='red', bins=20)
                
                ax.set_title(f'{caracteristica}')
                ax.set_xlabel('Valor')
                ax.set_ylabel('Frecuencia')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distribucion_caracteristicas_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado: {filename}")
        
        plt.show()
    
    def analisis_pca(self, guardar=True):
        """Realiza análisis de componentes principales"""
        if self.datos is None:
            print("❌ No hay datos cargados")
            return
        
        X = self.datos['caracteristicas']
        y = self.datos['etiquetas']
        
        # Normalizar datos
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Visualizar varianza explicada
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Varianza explicada por componente
        ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-')
        ax1.set_xlabel('Componente Principal')
        ax1.set_ylabel('Varianza Explicada')
        ax1.set_title('Varianza Explicada por Componente')
        ax1.grid(True)
        
        # Varianza acumulada
        ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                np.cumsum(pca.explained_variance_ratio_), 'ro-')
        ax2.set_xlabel('Número de Componentes')
        ax2.set_ylabel('Varianza Acumulada')
        ax2.set_title('Varianza Acumulada')
        ax2.grid(True)
        ax2.axhline(y=0.95, color='k', linestyle='--', label='95%')
        ax2.legend()
        
        plt.tight_layout()
        
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pca_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 PCA guardado: {filename}")
        
        plt.show()
        
        # Scatter plot 2D
        plt.figure(figsize=(10, 8))
        mosquitos_idx = y == 1
        no_mosquitos_idx = y == 0
        
        plt.scatter(X_pca[no_mosquitos_idx, 0], X_pca[no_mosquitos_idx, 1], 
                   alpha=0.6, c='blue', label='No-Mosquitos')
        plt.scatter(X_pca[mosquitos_idx, 0], X_pca[mosquitos_idx, 1], 
                   alpha=0.6, c='red', label='Mosquitos')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        plt.title('Análisis PCA - Primeros 2 Componentes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if guardar:
            filename = f"pca_scatter_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 PCA Scatter guardado: {filename}")
        
        plt.show()
        
        # Mostrar componentes más importantes
        print("\n🔍 COMPONENTES PRINCIPALES MÁS IMPORTANTES:")
        n_features = min(len(self.nombres_caracteristicas), X.shape[1])
        
        for i in range(min(3, len(pca.components_))):
            print(f"\nComponente {i+1} ({pca.explained_variance_ratio_[i]:.2%} varianza):")
            
            # Obtener las características más importantes
            importances = np.abs(pca.components_[i][:n_features])
            indices = np.argsort(importances)[::-1]
            
            for j in range(min(5, len(indices))):
                idx = indices[j]
                if idx < len(self.nombres_caracteristicas):
                    print(f"  {self.nombres_caracteristicas[idx]}: {pca.components_[i][idx]:.3f}")
    
    def matriz_correlacion(self, guardar=True):
        """Genera matriz de correlación de características"""
        if self.datos is None:
            print("❌ No hay datos cargados")
            return
        
        X = self.datos['caracteristicas']
        
        # Crear DataFrame con nombres de características
        n_features = min(len(self.nombres_caracteristicas), X.shape[1])
        df = pd.DataFrame(X[:, :n_features], columns=self.nombres_caracteristicas[:n_features])
        
        # Calcular matriz de correlación
        correlation_matrix = df.corr()
        
        # Visualizar
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlación'})
        plt.title('Matriz de Correlación de Características')
        plt.tight_layout()
        
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlacion_matrix_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Matriz de correlación guardada: {filename}")
        
        plt.show()
        
        # Mostrar correlaciones más fuertes
        print("\n🔗 CORRELACIONES MÁS FUERTES:")
        correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Solo correlaciones fuertes
                    correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlacion': corr
                    })
        
        # Ordenar por valor absoluto de correlación
        correlations.sort(key=lambda x: abs(x['correlacion']), reverse=True)
        
        for corr_data in correlations[:10]:  # Top 10
            print(f"  {corr_data['var1']} ↔ {corr_data['var2']}: {corr_data['correlacion']:.3f}")
    
    def analizar_importancia_caracteristicas(self):
        """Analiza la importancia de características usando el modelo entrenado"""
        if self.modelo_datos is None:
            print("❌ No hay modelo cargado")
            return
        
        modelo = self.modelo_datos['modelo']
        
        # Intentar obtener importancia de características
        importancia = None
        metodo = ""
        
        if hasattr(modelo, 'feature_importances_'):
            # Random Forest, Gradient Boosting
            importancia = modelo.feature_importances_
            metodo = "Feature Importances"
        elif hasattr(modelo, 'coef_'):
            # Modelos lineales, SVM
            if len(modelo.coef_.shape) == 1:
                importancia = np.abs(modelo.coef_)
            else:
                importancia = np.abs(modelo.coef_[0])
            metodo = "Coeficientes (valor absoluto)"
        
        if importancia is not None:
            print(f"\n🎯 IMPORTANCIA DE CARACTERÍSTICAS ({metodo}):")
            print("=" * 50)
            
            # Crear pares (importancia, nombre) y ordenar
            n_features = min(len(self.nombres_caracteristicas), len(importancia))
            importancia_nombres = list(zip(importancia[:n_features], 
                                         self.nombres_caracteristicas[:n_features]))
            importancia_nombres.sort(reverse=True)
            
            for i, (imp, nombre) in enumerate(importancia_nombres):
                print(f"  {i+1:2d}. {nombre:20} : {imp:.4f}")
            
            # Visualizar
            nombres = [item[1] for item in importancia_nombres]
            valores = [item[0] for item in importancia_nombres]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(nombres)), valores)
            plt.yticks(range(len(nombres)), nombres)
            plt.xlabel('Importancia')
            plt.title(f'Importancia de Características ({metodo})')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"importancia_caracteristicas_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Importancia guardada: {filename}")
            plt.show()
        else:
            print("❌ El modelo no soporta análisis de importancia de características")
    
    def curvas_roc_pr(self, datos_test=None):
        """Genera curvas ROC y Precision-Recall"""
        if self.modelo_datos is None or self.datos is None:
            print("❌ Se necesitan tanto modelo como datos")
            return
        
        modelo = self.modelo_datos['modelo']
        scaler = self.modelo_datos['scaler']
        
        X = self.datos['caracteristicas']
        y = self.datos['etiquetas']
        
        # Si no hay datos de test, usar todos los datos (no ideal, pero informativo)
        X_scaled = scaler.transform(X)
        
        # Obtener probabilidades
        if hasattr(modelo, 'predict_proba'):
            y_proba = modelo.predict_proba(X_scaled)[:, 1]
        else:
            print("❌ El modelo no soporta predicción de probabilidades")
            return
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
        
        # Visualizar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Curva ROC
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC)')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        ax2.plot(recall, precision, color='red', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"roc_pr_curves_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Curvas ROC/PR guardadas: {filename}")
        plt.show()
        
        print(f"\n📈 MÉTRICAS:")
        print(f"  AUC-ROC: {roc_auc:.3f}")
        print(f"  AUC-PR:  {pr_auc:.3f}")
    
    def generar_reporte_completo(self):
        """Genera un reporte completo de análisis"""
        print("📋 GENERANDO REPORTE COMPLETO...")
        print("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio para reportes
        reporte_dir = f"reporte_mosquitos_{timestamp}"
        os.makedirs(reporte_dir, exist_ok=True)
        
        print(f"📁 Directorio del reporte: {reporte_dir}")
        
        # Cambiar directorio de trabajo temporalmente
        original_dir = os.getcwd()
        os.chdir(reporte_dir)
        
        try:
            print("  📊 Generando estadísticas básicas...")
            df = self.estadisticas_basicas()
            
            print("  📈 Generando distribuciones...")
            self.visualizar_distribucion_caracteristicas()
            
            print("  🔍 Realizando análisis PCA...")
            self.analisis_pca()
            
            print("  🔗 Generando matriz de correlación...")
            self.matriz_correlacion()
            
            if self.modelo_datos:
                print("  🎯 Analizando importancia de características...")
                self.analizar_importancia_caracteristicas()
                
                print("  📈 Generando curvas ROC/PR...")
                self.curvas_roc_pr()
            
            # Generar resumen en texto
            with open("resumen_analisis.txt", "w", encoding='utf-8') as f:
                f.write(f"REPORTE DE ANÁLISIS - DETECTOR DE MOSQUITOS ML\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                if self.datos:
                    X = self.datos['caracteristicas']
                    y = self.datos['etiquetas']
                    
                    f.write("ESTADÍSTICAS DE DATOS:\n")
                    f.write(f"  Total muestras: {len(X)}\n")
                    f.write(f"  Mosquitos: {sum(y)} ({sum(y)/len(y)*100:.1f}%)\n")
                    f.write(f"  No-mosquitos: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)\n")
                    f.write(f"  Características: {X.shape[1]}\n\n")
                
                f.write("ARCHIVOS GENERADOS:\n")
                for archivo in os.listdir("."):
                    if archivo.endswith(('.png', '.jpg')):
                        f.write(f"  - {archivo}\n")
            
            print(f"\n✅ Reporte completo generado en: {reporte_dir}")
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")
        
        finally:
            os.chdir(original_dir)

def main():
    analizador = AnalizadorDatosMosquitos()
    
    print("🔬 HERRAMIENTAS DE ANÁLISIS - DETECTOR ML MOSQUITOS")
    print("=" * 55)
    
    while True:
        print("\n📋 MENÚ DE ANÁLISIS:")
        print("1. Cargar datos de entrenamiento")
        print("2. Cargar modelo entrenado")
        print("3. Estadísticas básicas")
        print("4. Visualizar distribuciones")
        print("5. Análisis PCA")
        print("6. Matriz de correlación")
        print("7. Importancia de características")
        print("8. Curvas ROC y Precision-Recall")
        print("9. Generar reporte completo")
        print("10. Salir")
        
        opcion = input("\nSelecciona una opción (1-10): ").strip()
        
        if opcion == "1":
            archivos_datos = [f for f in os.listdir('.') if f.startswith('datos_mosquitos_') and f.endswith('.pkl')]
            
            if not archivos_datos:
                print("❌ No se encontraron archivos de datos")
                continue
            
            print("\n📁 Archivos disponibles:")
            for i, archivo in enumerate(archivos_datos):
                print(f"  {i+1}. {archivo}")
            
            seleccion = input("Selecciona archivo (número): ").strip()
            if seleccion.isdigit() and 1 <= int(seleccion) <= len(archivos_datos):
                archivo_seleccionado = archivos_datos[int(seleccion)-1]
                analizador.cargar_datos(archivo_seleccionado)
        
        elif opcion == "2":
            archivos_modelo = [f for f in os.listdir('.') if f.startswith('modelo_mosquito_') and f.endswith('.pkl')]
            
            if not archivos_modelo:
                print("❌ No se encontraron modelos")
                continue
            
            print("\n🤖 Modelos disponibles:")
            for i, archivo in enumerate(archivos_modelo):
                print(f"  {i+1}. {archivo}")
            
            seleccion = input("Selecciona modelo (número): ").strip()
            if seleccion.isdigit() and 1 <= int(seleccion) <= len(archivos_modelo):
                archivo_seleccionado = archivos_modelo[int(seleccion)-1]
                analizador.cargar_modelo(archivo_seleccionado)
        
        elif opcion == "3":
            analizador.estadisticas_basicas()
        
        elif opcion == "4":
            analizador.visualizar_distribucion_caracteristicas()
        
        elif opcion == "5":
            analizador.analisis_pca()
        
        elif opcion == "6":
            analizador.matriz_correlacion()
        
        elif opcion == "7":
            analizador.analizar_importancia_caracteristicas()
        
        elif opcion == "8":
            analizador.curvas_roc_pr()
        
        elif opcion == "9":
            analizador.generar_reporte_completo()
        
        elif opcion == "10":
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()