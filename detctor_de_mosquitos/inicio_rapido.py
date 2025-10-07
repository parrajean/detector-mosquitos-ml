#!/usr/bin/env python3
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
    
    print("\nOpciones disponibles:")
    for key, (desc, _) in opciones.items():
        print(f"  {key}. {desc}")
    
    seleccion = input("\nSelecciona una opcion (1-4): ").strip()
    
    if seleccion in opciones:
        _, comando = opciones[seleccion]
        print(f"\nEjecutando: {comando}")
        os.system(comando)
    else:
        print("Opcion invalida")

if __name__ == "__main__":
    main()
