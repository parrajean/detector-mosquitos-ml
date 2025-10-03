import os

print("📁 VERIFICACIÓN DE ARCHIVOS")
print("="*50)

# Directorio actual
print(f"\n📍 Directorio actual: {os.getcwd()}")

# Archivos necesarios
archivos_necesarios = [
    'detector_mejorado.py',
    'app_web.py',
    'templates/index.html'
]

print("\n✅ Archivos encontrados:")
print("❌ Archivos faltantes:")
print()

for archivo in archivos_necesarios:
    existe = os.path.exists(archivo)
    simbolo = "✅" if existe else "❌"
    print(f"{simbolo} {archivo}")
    if existe:
        print(f"   Ruta: {os.path.abspath(archivo)}")

# Buscar detector_mejorado.py en subcarpetas
print("\n🔍 Buscando detector_mejorado.py...")
for root, dirs, files in os.walk('.'):
    if 'detector_mejorado.py' in files:
        print(f"   Encontrado en: {os.path.join(root, 'detector_mejorado.py')}")

print("\n" + "="*50)