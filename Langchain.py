import time
import requests
import google.generativeai as genai
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv



# Configuración de logging simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar las variables del archivo .env
load_dotenv(".env", override=True)  # Si .env está en la misma carpeta


# Obtener las claves desde el entorno
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurar APIs
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini configurado")
except Exception as e:
    logger.error(f"❌ Error configurando Gemini: {e}")

try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI configurado")
except Exception as e:
    logger.error(f"❌ Error configurando OpenAI: {e}")

logger.info("APIs configuradas: Groq, Gemini, OpenAI")

# Variables globales
retriever = None
historial = []

# Prompt simple
prompt_template = """
**Instrucciones Generales:**
Actúa como un asistente experto en análisis de documentos y respuestas basadas en datos. Tu objetivo es proporcionar respuestas claras, precisas, completas y bien estructuradas basadas exclusivamente en el contexto proporcionado y el historial de la conversación. Sigue estas directrices estrictamente:

1. **Uso del Contexto y Historial:**
   - Utiliza únicamente la información del **contexto** y el **historial** para responder.
   - Si el contexto no contiene información suficiente, responde: "No hay suficiente información en el contexto para responder esta pregunta."
   - Integra fragmentos relevantes del contexto de manera natural, citándolos o parafraseándolos según sea necesario.
   - Considera el historial para mantener coherencia en las respuestas, especialmente si la pregunta se refiere a interacciones previas.

2. **Claridad y Precisión:**
   - Responde en un lenguaje claro, profesional y adaptado al nivel de comprensión del usuario.
   - Evita jerga técnica a menos que la pregunta lo requiera; en ese caso, explica los términos.
   - Si la pregunta es ambigua, identifica la ambigüedad y sugiere una reformulación específica para aclararla. Por ejemplo: "Tu pregunta podría referirse a X o Y. ¿Podrías precisar si te refieres a X o Y?"

3. **Estructura de la Respuesta:**
   - Estructura la respuesta en secciones claras (si aplica) usando encabezados como "Respuesta Principal", "Detalles Adicionales" o "Ejemplo".
   - Usa listas, viñetas o numeración para mejorar la legibilidad cuando sea necesario.
   - Proporciona ejemplos concretos si la pregunta lo permite y el contexto lo soporta.
   - Si la pregunta requiere un análisis profundo, divide la respuesta en pasos lógicos.

4. **Manejo de Preguntas Complejas:**
   - Si la pregunta tiene múltiples partes, responde cada una por separado con un encabezado claro.
   - Si el contexto contiene información contradictoria, señala la contradicción y ofrece la interpretación más plausible basada en los datos disponibles.
   - Para preguntas abiertas, ofrece una respuesta completa pero concisa, priorizando la información más relevante.

5. **Tono y Ética:**
   - Mantén un tono neutral, respetuoso y profesional en todo momento.
   - Evita cualquier sesgo, suposición o información no verificada.
   - Si la pregunta toca temas sensibles, responde con sensibilidad y enfócate en hechos objetivos.

6. **Multilingüismo:**
   - Responde en el idioma de la pregunta (en este caso, español) a menos que se indique lo contrario.
   - Si se solicita una traducción o respuesta en otro idioma, proporciónala junto con la respuesta en español.

7. **Formato de Salida:**
   - Usa un formato markdown claro para mejorar la legibilidad (por ejemplo, **negritas** para énfasis, *cursivas* para citas, o listas para puntos clave).
   - Si la respuesta es extensa, incluye un resumen inicial breve antes de los detalles.
   - Termina con una conclusión o recomendación si la pregunta lo permite.

**Contexto:**
{context}

**Historial de Conversación:**
{historial}

**Pregunta Actual:**
{question}

**Respuesta:**
- **Resumen**: [Proporciona un resumen breve de la respuesta, si aplica]
- **Respuesta Principal**: [Desarrollo completo de la respuesta, integrando el contexto y el historial]
- **Detalles Adicionales** (opcional): [Información complementaria o aclaraciones]
- **Conclusión** (opcional): [Cierre o recomendación basada en la respuesta]

**Nota**: Si no puedes responder completamente debido a limitaciones en el contexto, explica por qué y sugiere cómo obtener más información.
"""

# Preguntas preestablecidas con ground truth
PREGUNTAS_Y_GROUND_TRUTH = [
    {
        "question": "¿Qué es Q-Learning y cómo funciona?",
        "ground_truth": "Q-Learning es un algoritmo de aprendizaje por refuerzo sin modelo que aprende la función de valor Q(s,a) que representa la recompensa esperada de tomar la acción 'a' en el estado 's' y seguir la política óptima. Funciona mediante la actualización iterativa de una tabla Q usando la ecuación Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]."
    },
    {
        "question": "¿Cuál es la ecuación de Bellman en Q-Learning?",
        "ground_truth": "La ecuación de Bellman en Q-Learning es: Q(s,a) = R(s,a) + γ * max[Q(s',a')] donde Q(s,a) es el valor Q del estado s y acción a, R(s,a) es la recompensa inmediata, γ es el factor de descuento, y max[Q(s',a')] es el máximo valor Q del siguiente estado s'."
    },
    {
        "question": "¿Qué significa la tasa de aprendizaje en Q-Learning?",
        "ground_truth": "La tasa de aprendizaje (α) en Q-Learning controla qué tan rápido el agente actualiza sus valores Q. Un valor alto (cerca de 1) hace que el agente aprenda rápidamente pero sea inestable, mientras que un valor bajo (cerca de 0) hace el aprendizaje más estable pero lento. Típicamente se usa α entre 0.1 y 0.5."
    },
    {
        "question": "¿Cuál es la diferencia entre exploración y explotación?",
        "ground_truth": "Exploración significa que el agente prueba acciones aleatorias para descubrir nuevas estrategias y evitar quedarse en óptimos locales. Explotación significa que el agente elige la mejor acción conocida basada en su experiencia actual. El balance entre ambas se maneja típicamente con estrategias como epsilon-greedy."
    },
    {
        "question": "¿Cómo se actualiza la tabla Q?",
        "ground_truth": "La tabla Q se actualiza usando la regla: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)] donde α es la tasa de aprendizaje, r es la recompensa recibida, γ es el factor de descuento, y el término entre corchetes es el error temporal difference (TD)."
    },
    {
        "question": "¿Qué es el factor de descuento?",
        "ground_truth": "El factor de descuento (γ) determina la importancia de las recompensas futuras versus las inmediatas. Un valor cercano a 0 hace que el agente sea miope (solo considera recompensas inmediatas), mientras que un valor cercano a 1 hace que considere igualmente las recompensas a largo plazo. Típicamente γ está entre 0.9 y 0.99."
    },
    {
        "question": "¿Cuáles son las ventajas del Q-Learning?",
        "ground_truth": "Las ventajas del Q-Learning incluyen: no requiere modelo del entorno, garantiza convergencia a la política óptima bajo ciertas condiciones, es simple de implementar, funciona con espacios de estados y acciones discretos, y puede manejar problemas estocásticos."
    },
    {
        "question": "¿En qué problemas se aplica Q-Learning?",
        "ground_truth": "Q-Learning se aplica en: videojuegos (como Atari), navegación robótica, control de tráfico, sistemas de recomendación, trading financiero, optimización de recursos, gestión de inventarios, y cualquier problema de toma de decisiones secuencial con recompensas."
    },
    {
        "question": "¿Qué es la política epsilon-greedy?",
        "ground_truth": "La política epsilon-greedy es una estrategia que balancea exploración y explotación. Con probabilidad (1-ε) elige la mejor acción conocida (explotación) y con probabilidad ε elige una acción aleatoria (exploración). ε típicamente decrece durante el entrenamiento desde ~1.0 hasta ~0.1."
    },
    {
        "question": "¿Cómo se evalúa un agente Q-Learning?",
        "ground_truth": "Un agente Q-Learning se evalúa mediante: recompensa acumulada promedio por episodio, tasa de convergencia a la política óptima, estabilidad del aprendizaje, tiempo de entrenamiento requerido, y rendimiento comparado con otros algoritmos. También se usan métricas como la pérdida TD y la exploración efectiva."
    }
]

# 📊 DATOS PARA EXPORTAR JSON
datos_para_json = []

def cargar_pdf(pdf_path):
    """Cargar y procesar PDF"""
    global retriever
    
    print("📄 Cargando PDF...")
    
    # Cargar PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Crear embeddings y vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity")
    
    print(f"✅ PDF procesado: {len(docs)} páginas, {len(splits)} chunks")

def consultar_groq(prompt, max_reintentos=3):
    """Consultar API de Groq con reintentos y rate limiting"""
    for intento in range(max_reintentos):
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(url, headers=headers, json=body, timeout=30)
            
            # Si es rate limit (429), espera más tiempo
            if response.status_code == 429:
                tiempo_espera = (intento + 1) * 10  # 10, 20, 30 segundos
                print(f"⏳ Rate limit alcanzado. Esperando {tiempo_espera} segundos...")
                time.sleep(tiempo_espera)
                continue
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            if intento < max_reintentos - 1:
                tiempo_espera = (intento + 1) * 5  # 5, 10 segundos
                print(f"⚠️ Error en intento {intento + 1}. Reintentando en {tiempo_espera}s...")
                time.sleep(tiempo_espera)
            else:
                return f"❌ Error Groq después de {max_reintentos} intentos: {str(e)[:100]}"
        except Exception as e:
            return f"❌ Error Groq: {str(e)[:100]}"
    
    return "❌ Error Groq: Máximo de reintentos alcanzado"

def consultar_gemini(prompt):
    """Consultar API de Gemini"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error Gemini: {str(e)[:100]}"

def consultar_openai(prompt):
    """Consultar API de OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error OpenAI: {str(e)[:100]}"

def hacer_pregunta_json(pregunta_data, modelo="groq"):
    """Hacer una pregunta al sistema RAG y recopilar datos para JSON"""
    global retriever, historial, datos_para_json
    
    if not retriever:
        return {"error": "PDF no cargado"}
    
    pregunta = pregunta_data["question"]
    ground_truth = pregunta_data["ground_truth"]
    
    # Obtener contexto relevante (documentos recuperados)
    docs = retriever.get_relevant_documents(pregunta)
    contexts = [doc.page_content for doc in docs[:3]]  # Top 3 documentos más relevantes
    
    # Preparar prompt
    historial_str = "\n".join([f"P: {p}\nR: {r[:100]}..." for p, r in historial[-3:]])
    prompt = prompt_template.format(
        context="\n".join(contexts[:2000]),  # Limitar contexto
        historial=historial_str,
        question=pregunta
    )
    
    # Obtener respuesta del modelo especificado
    if modelo == "groq":
        respuesta = consultar_groq(prompt)
    elif modelo == "gemini":
        respuesta = consultar_gemini(prompt)
    elif modelo == "openai":
        respuesta = consultar_openai(prompt)
    else:
        respuesta = consultar_groq(prompt)  # Default
    
    # Limpiar respuesta si hay error
    if respuesta.startswith("❌"):
        respuesta = "Error en la consulta al modelo"
    
    # Crear objeto para JSON
    pregunta_objeto = {
        "id": len(datos_para_json) + 1,
        "question": pregunta,
        "contexts": contexts,
        "answer": respuesta,
        "ground_truth": ground_truth,
        "modelo": modelo,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "num_contextos": len(contexts),
            "longitud_respuesta": len(respuesta),
            "longitud_pregunta": len(pregunta)
        }
    }
    
    # Agregar a datos para JSON
    datos_para_json.append(pregunta_objeto)
    
    # Guardar en historial
    historial.append((pregunta, respuesta[:200]))
    if len(historial) > 5:
        historial.pop(0)
    
    return {
        "pregunta": pregunta,
        "contextos": contexts,
        "respuesta": respuesta,
        "ground_truth": ground_truth
    }

def exportar_json(filename="respuestas_rag.json", modelo="groq"):
    """Exportar datos al formato JSON"""
    
    if not datos_para_json:
        print("❌ No hay datos para exportar")
        return
    
    # Filtrar por modelo si se especifica
    if modelo != "todos":
        datos_filtrados = [d for d in datos_para_json if d["modelo"] == modelo]
    else:
        datos_filtrados = datos_para_json
    
    if not datos_filtrados:
        print(f"❌ No hay datos para el modelo {modelo}")
        return
    
    # Crear estructura JSON completa
    datos_json = {
        "metadata": {
            "total_preguntas": len(datos_filtrados),
            "modelos_usados": list(set([d["modelo"] for d in datos_filtrados])),
            "fecha_exportacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        },
        "preguntas_y_respuestas": datos_filtrados
    }
    
    # Guardar JSON
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(datos_json, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Datos exportados a {filename}")
        print(f"📊 Registros exportados: {len(datos_filtrados)}")
        print(f"🤖 Modelo(s) usado(s): {', '.join(datos_json['metadata']['modelos_usados'])}")
        
        return datos_json
        
    except Exception as e:
        print(f"❌ Error al exportar JSON: {str(e)}")
        return None

def ejecutar_preguntas_json(modelo="groq"):
    """Ejecutar las 10 preguntas y recopilar datos para JSON"""
    print(f"\n🧠 Ejecutando 10 preguntas con modelo: {modelo.upper()}")
    print("=" * 60)
    
    resultados_totales = []
    
    for i, pregunta_data in enumerate(PREGUNTAS_Y_GROUND_TRUTH, 1):
        print(f"\n⏳ Procesando pregunta {i}/10...")
        print(f"❓ {pregunta_data['question']}")
        
        # Hacer pregunta
        resultado = hacer_pregunta_json(pregunta_data, modelo)
        
        if "error" in resultado:
            print(f"❌ {resultado['error']}")
            continue
        
        # Mostrar respuesta resumida
        respuesta_corta = resultado["respuesta"][:200] + "..." if len(resultado["respuesta"]) > 200 else resultado["respuesta"]
        print(f"🤖 Respuesta: {respuesta_corta}")
        
        resultados_totales.append(resultado)
        
        # Pausa entre preguntas
        time.sleep(1)
    
    return resultados_totales

def mostrar_resumen():
    """Mostrar resumen de datos recopilados"""
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE DATOS RECOPILADOS")
    print('='*60)
    print(f"✅ Total de registros: {len(datos_para_json)}")
    
    if datos_para_json:
        modelos = set([d["modelo"] for d in datos_para_json])
        print(f"🤖 Modelos usados: {', '.join(modelos)}")
        print(f"📝 Preguntas únicas: {len(set([d['question'] for d in datos_para_json]))}")
        print(f"📄 Promedio de contextos por pregunta: {sum([d['metadata']['num_contextos'] for d in datos_para_json]) / len(datos_para_json):.1f}")

def main():
    """Función principal"""
    print("🧠 Sistema RAG con Exportación JSON")
    print("=" * 60)
    
    # Cargar PDF
    pdf_path = "docs/PDF_IA.pdf"
    
    try:
        cargar_pdf(pdf_path)
        
        # Seleccionar modelo
        print("\n🤖 Selecciona el modelo a usar:")
        print("1. Groq (llama3-70b-8192)")
        print("2. Gemini (gemini-1.5-flash)")
        print("3. OpenAI (gpt-3.5-turbo)")
        print("4. Todos los modelos")
        
        opcion = input("\n🎯 Opción (1-4, default=1): ").strip()
        
        modelos_map = {
            "1": "groq",
            "2": "gemini", 
            "3": "openai",
            "4": "todos"
        }
        
        modelo_seleccionado = modelos_map.get(opcion, "groq")
        
        if modelo_seleccionado == "todos":
            # Ejecutar con todos los modelos
            for modelo in ["groq", "gemini", "openai"]:
                print(f"\n🚀 Ejecutando con {modelo.upper()}...")
                ejecutar_preguntas_json(modelo)
        else:
            # Ejecutar con modelo seleccionado
            ejecutar_preguntas_json(modelo_seleccionado)
        
        # Mostrar resumen
        mostrar_resumen()
        
        # Exportar JSON
        print(f"\n💾 ¿Exportar datos a JSON?")
        exportar = input("📝 (s/n, default=s): ").strip().lower()
        
        if exportar in ['', 's', 'si', 'y', 'yes']:
            if modelo_seleccionado == "todos":
                # Exportar por modelo
                for modelo in ["groq", "gemini", "openai"]:
                    filename = f"respuestas_rag_{modelo}1.json"
                    exportar_json(filename, modelo)
                
                # Exportar todos juntos
                exportar_json("respuestas_rag_todos.json", "todos")
            else:
                filename = f"respuestas_rag_{modelo_seleccionado}.json"
                exportar_json(filename, modelo_seleccionado)
        
        print(f"\n🎉 ¡Proceso completado!")
        print(f"📁 Archivos JSON listos con todas las respuestas")
        
    except FileNotFoundError:
        print("❌ Error: No se encontró 'docs/PDF_IA.pdf'")
        print("💡 Coloca tu PDF en la carpeta 'docs/' con el nombre 'PDF_IA.pdf'")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()