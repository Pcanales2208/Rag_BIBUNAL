# 📚 RAG-BIBUNAL: Agente Inteligente de Consulta Bibliográfica con APIs de LLM

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** para responder preguntas académicas a partir de documentos PDF de la Biblioteca de la Universidad Nacional de Colombia, Sede La Paz. Utiliza modelos de lenguaje avanzados alojados vía API (Gemini, OpenAI, Groq) y realiza recuperación semántica con embeddings y un vectorstore local (ChromaDB).

---

## 🚀 Características

- ✅ Recuperación semántica precisa desde documentos PDF
- ✅ Interfaz web con **Gradio**
- ✅ Selección de LLM (Groq, OpenAI, Gemini)
- ✅ Integración con **ChromaDB**, **HuggingFace Transformers**, y más
- ✅ Evaluación con métricas como **Factualidad** y **Relevancia** (via [RAGAS](https://github.com/explodinggradients/ragas))

---

## 🧠 Tecnologías usadas

| Componente                  | Tecnología / Herramienta         |
|----------------------------|----------------------------------|
| Embeddings                 | `sentence-transformers/all-MiniLM-L6-v2` |
| Vectorstore                | `Chroma`                         |
| Lectura de PDFs            | `PyPDFLoader` de `langchain`     |
| División de texto          | `RecursiveCharacterTextSplitter`|
| LLMs (vía API)             | Gemini, OpenAI, Groq (LLaMA 3)   |
| Evaluación del sistema     | RAGAS                            |

