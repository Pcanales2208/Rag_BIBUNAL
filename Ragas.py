import os
import json
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
import pandas as pd
from dotenv import load_dotenv      

load_dotenv(".env", override=True)
# 1. Configura tu API Key de OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Cargar el archivo JSON con los ejemplos
with open("respuestas_rag_openai2.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 3. Extraer solo los datos necesarios para RAGAS
data = []
for item in json_data["preguntas_y_respuestas"]:
    data.append({
        "question": item["question"],
        "contexts": item["contexts"],
        "answer": item["answer"],
        "ground_truth": item["ground_truth"]
    })

print(f"📊 Datos cargados: {len(data)} registros")

# 4. Convertir a un Dataset compatible con RAGAS
dataset = Dataset.from_list(data)

# 5. Evaluar con las métricas seleccionadas
print("🔄 Evaluando con RAGAS...")
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision, 
        context_recall
    ]
)

print("✅ Resultados de la evaluación:")
print(result)

result.to_pandas().to_csv("resultados_ragasOpenai2.csv", index=False)