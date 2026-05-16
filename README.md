# 🎓 KonsultaBot — Asistente de Reglamentos Institucionales
### Fundación Universitaria Konrad Lorenz

> **Materia:** Desarrollo de Aplicaciones con IA  
> **Estudiante:** Laura Valentina López García  
> **Proyecto:** Desarrollo de un Asistente Experto basado en RAG

---

## ⚙️ Instalación y configuración

```bash
# 1. Clonar el repositorio
git clone https://github.com/valentinalopezgrc/konsulta-bot.git
cd konsulta-bot

# 2. Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate         # Mac/Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key
cp .env.example .env
# Editar .env y agregar:
# GENAI_API_KEY=tu_clave_aqui

# 5. Ejecutar la interfaz gráfica (recomendado)
python konsulta_bot_gui.py

# O ejecutar en modo consola
python konsulta_bot_rag.py
```

> ⚠️ **Nota sobre la API Key:** Genera tu clave desde un **proyecto nuevo** en [Google AI Studio](https://aistudio.google.com/apikey). El free tier tiene límite de embeddings por día (~1500 embeddings). Si la cuota se agota, el bot devolverá error 429. Se recomienda usar una API key con cuota disponible.

> ℹ️ La primera vez que se ejecuta, construye la base vectorial automáticamente llamando a la API de Gemini Embeddings (tarda ~15 minutos por los rate limits del plan gratuito). Las siguientes ejecuciones cargan la base existente directamente.

---

## ¿Qué es KonsultaBot?

**KonsultaBot** es un asistente conversacional especializado en los **reglamentos institucionales de la Fundación Universitaria Konrad Lorenz**. Está diseñado para ayudar a estudiantes y docentes a consultar artículos de los reglamentos, entender sus derechos y obligaciones, y saber a qué instancia acudir según su situación académica.

El nombre **KonsultaBot** es un juego de palabras entre **Konrad** y **Consulta**.

---

## 🚀 Entrega Final — RAG con Gemini Embeddings + GUI Gradio

### Descripción
Versión final del sistema RAG con dos mejoras principales respecto al Avance 2: migración del modelo de embeddings de SentenceTransformers a **Gemini Embedding 001** para mejor comprensión del español jurídico, y adición de una **interfaz gráfica web** construida con Gradio.

### Flujo del sistema RAG

```
PDFs → CARGA (pypdf)
     → CHUNKING (RecursiveCharacterTextSplitter, 800 chars, overlap 100)
     → EMBEDDINGS (Gemini Embedding 001 via API — gemini-embedding-001)
     → BASE VECTORIAL (ChromaDB, similitud coseno)
     → CONSULTA del usuario
     → RETRIEVAL (TOP-7 chunks más similares)
     → PROMPT AUMENTADO (system prompt + few-shot + contexto recuperado)
     → GEMINI 2.5 Flash
     → JSON estructurado → GUI Gradio
```

### Reglamentos indexados

| Reglamento | Páginas | Caracteres |
|-----------|---------|------------|
| Reglamento Académico de Pregrado | 52 | 109,733 |
| Reglamento Académico Institucional | 64 | 117,256 |
| Reglamento Docente | 35 | 62,060 |
| Reglamento Académico de Posgrado | 34 | 80,285 |
| **Total** | **185** | **369,334** |

### Componentes del pipeline

| Componente | Tecnología | Descripción |
|-----------|-----------|-------------|
| Extracción PDF | `pypdf` | Lee y extrae texto de los PDFs |
| Chunking | `langchain-text-splitters` | Divide el texto en fragmentos de 800 chars con overlap de 100 |
| Embeddings | `Gemini Embedding 001` | Vectoriza los chunks vía API de Google — mejor semántica en español jurídico |
| Base vectorial | `ChromaDB` | Almacena y consulta los embeddings por similitud coseno |
| LLM | `Google Gemini 2.5 Flash` | Genera la respuesta final en JSON |
| Interfaz | `Gradio` | Interfaz web con historial de conversación |

### Configuración del sistema

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 800 | Tamaño de cada fragmento en caracteres |
| `CHUNK_OVERLAP` | 100 | Solapamiento entre fragmentos consecutivos |
| `TOP_K` | 7 | Número de chunks recuperados por consulta |
| `COLLECTION` | konsulta_reglamentos | Nombre de la colección en ChromaDB |
| `MODELO_EMBEDDINGS` | gemini-embedding-001 | Modelo de embeddings vía API de Google |
| `LLM` | gemini-2.5-flash | Modelo generativo para respuestas |

### Estructuración del prompt

**System Prompt:**
```
Eres KonsultaBot, un asistente experto en los Reglamentos Académicos
de la Fundación Universitaria Konrad Lorenz.
Responde ÚNICAMENTE con base en el contexto recuperado entre
<CONTEXTO_RAG> y </CONTEXTO_RAG>.
Nunca inventes artículos, porcentajes ni fechas que no estén en ese contexto.
Responde SIEMPRE con un JSON válido con exactamente estos 5 campos,
sin texto fuera del JSON.
```

**Few-Shot (5 ejemplos):**

| Pregunta | Artículo | Tipo |
|---------|---------|------|
| ¿Cuántas fallas me reprueban? | Art. 43 Par.1 | Inasistencias pregrado |
| Perdí 4 materias, ¿qué pasa? | Art. 79 | Pérdida de cupo |
| ¿Puedo pedir supletorio? | Art. 65 y 66 | Prueba especial |
| ¿Cuántas fallas en posgrado? | Art. 31 Par.1 | Inasistencias posgrado |
| ¿Cuál es el horario de la cafetería? | null | Fuera de dominio |

**Formato de salida JSON:**
```json
{
  "articulo": "Art. XX o null",
  "respuesta": "Explicación clara para el estudiante",
  "cita_textual": "Fragmento literal del reglamento o null",
  "accion_recomendada": "Qué debe hacer / a quién acudir",
  "advertencia": "Riesgo académico relevante o null"
}
```

### Resultados de evaluación (10 preguntas)

| P# | Reglamento | Tipo | Resultado |
|----|-----------|------|-----------|
| P1 | Institucional | Textual directa | ✅ Correcto |
| P2 | Pregrado | Textual directa | ✅ Correcto |
| P3 | Institucional | Vocabulario diferente | ✅ Correcto |
| P4 | Posgrado | Vocabulario diferente | ✅ Correcto |
| P5 | Institucional | Multi-chunk | ⚠️ Parcial |
| P6 | Institucional | Multi-chunk | ✅ Correcto |
| P7 | Posgrado | Multi-chunk | ✅ Correcto |
| P8 | Docente | Anti-alucinación | ✅ Correcto |
| P9 | Docente | Anti-alucinación | ✅ Correcto |
| P10 | Ninguno | Anti-alucinación | ✅ Correcto |

**Resultado global: 9/10 correctas**

---

## 📁 Estructura del repositorio

```
konsulta-bot/
│
├── Avance 1/
│   ├── Capturas de pantalla Avance 1/
│   └── konsulta_bot.py
├── Capturas de pantalla Avance 2/
├── pdfs/
│   ├── reglamento-academico-de-pregrado.pdf
│   ├── reglamento-academico-institucional.pdf
│   ├── reglamento-docente.pdf
│   └── reglamento-academico-de-posgrado.pdf
├── .env.example
├── .gitignore
├── konsulta_bot_gui.py
├── konsulta_bot_rag.py
├── README.md
└── requirements.txt
```

---

## 📦 Dependencias

```
chromadb>=0.5.0
langchain-text-splitters
langchain-google-genai
colorama
google-genai>=1.0.0
pypdf>=4.0.0
python-dotenv>=1.0.0
gradio>=6.0.0
```

---

## 🛠️ Tecnologías usadas

- **Python 3.10+**
- **Google Gemini 2.5 Flash** via `google-genai` — modelo generativo
- **Google Gemini Embedding 001** via `langchain-google-genai` — embeddings semánticos
- **pypdf** — extracción de texto de PDFs
- **ChromaDB** — base de datos vectorial persistente
- **LangChain Text Splitters** — chunking inteligente por separadores
- **Gradio** — interfaz web conversacional
- **colorama** — interfaz de consola con colores
- **Prompt Engineering**: System Prompt + Few-Shot + XML + JSON output

---

## 📎 Evolución del proyecto

### Avance 1 — Prompt Stuffing (`Avance 1/konsulta_bot.py`)

El reglamento completo se extrae del PDF y se inyecta directamente en el prompt como contexto en cada consulta.

| Técnica | Descripción |
|---------|-------------|
| **System Prompt estructurado** | 6 secciones: ROL, TAREA, CONTEXTO, RESTRICCIONES, FORMATO, EJEMPLOS |
| **Few-Shot Prompting** | 5 ejemplos de pares pregunta/respuesta |
| **Delimitadores XML** | Etiquetas para separar comportamientos |
| **Formato JSON** | Salida estructurada con 5 campos fijos |
| **Prompt Stuffing** | Texto completo del PDF inyectado como contexto |

### Comparativa entre versiones

| | Avance 1 | Avance 2 | Entrega Final |
|--|---------|---------|--------------|
| Contexto | Todo el reglamento en el prompt | 7 chunks más relevantes | 7 chunks más relevantes |
| Documentos | 1 PDF | 4 PDFs | 4 PDFs |
| Embeddings | No aplica | SentenceTransformers (local) | Gemini Embedding 001 (API) |
| Base vectorial | No aplica | ChromaDB | ChromaDB |
| Interfaz | Consola | Consola | Gradio (web) |
| Precisión | ~60% | 6/10 | 9/10 |
| Escalabilidad | Limitada por tokens | Alta | Alta |