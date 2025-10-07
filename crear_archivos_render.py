import os

print("🔧 CREANDO ARCHIVOS PARA RENDER")
print("="*60)
print(f"📁 Directorio actual: {os.getcwd()}")

# 1. requirements.txt
print("\n1️⃣ Creando requirements.txt...")
requirements = """Flask==3.0.0
opencv-python-headless==4.8.1.78
numpy==1.26.2
gunicorn==21.2.0
Pillow==10.1.0
"""

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements)

if os.path.exists('requirements.txt'):
    print("✅ requirements.txt creado")
    with open('requirements.txt', 'r') as f:
        print("Contenido:")
        print(f.read())
else:
    print("❌ Error creando requirements.txt")

# 2. runtime.txt
print("\n2️⃣ Creando runtime.txt...")
with open('runtime.txt', 'w', encoding='utf-8') as f:
    f.write('python-3.11.4\n')

if os.path.exists('runtime.txt'):
    print("✅ runtime.txt creado")
else:
    print("❌ Error creando runtime.txt")

# 3. Procfile
print("\n3️⃣ Creando Procfile...")
with open('Procfile', 'w', encoding='utf-8') as f:
    f.write('web: gunicorn app_web:app --bind 0.0.0.0:$PORT\n')

if os.path.exists('Procfile'):
    print("✅ Procfile creado")
else:
    print("❌ Error creando Procfile")

# 4. Listar archivos creados
print("\n📋 Archivos en el directorio:")
for archivo in os.listdir('.'):
    if not archivo.startswith('.'):
        print(f"   - {archivo}")

print("\n" + "="*60)
print("✅ ARCHIVOS CREADOS")
print("="*60)

print("\n📋 PRÓXIMOS PASOS:")
print("\n1. Verificar archivos:")
print("   dir                  (Windows)")
print("   ls                   (Linux/Mac)")

print("\n2. Subir a GitHub:")
print("   git add .")
print("   git commit -m 'Add: Archivos de configuración'")
print("   git push")

print("\n3. En Render:")
print("   Manual Deploy → Deploy latest commit")

print("\n💡 Si git no detecta cambios:")
print("   git add -f requirements.txt runtime.txt Procfile")
print("   git commit -m 'Force add config files'")
print("   git push")