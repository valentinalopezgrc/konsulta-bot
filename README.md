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

# 5. Ejecutar Avance 2 — RAG (versión actual)
python konsulta_bot_rag.py
```

> ⚠️ **Nota sobre la API Key:** Genera tu clave desde un **proyecto nuevo** en [Google AI Studio](https://aistudio.google.com/apikey). El free tier tiene límite de embeddings por día. Si la cuota se agota, el bot devolverá error 429.

> ℹ️ La primera vez que se ejecuta, descarga el modelo de embeddings (~470MB) y construye la base vectorial automáticamente. Las siguientes ejecuciones cargan la base existente directamente.

---

## ¿Qué es KonsultaBot?

**KonsultaBot** es un asistente conversacional especializado en los **reglamentos institucionales de la Fundación Universitaria Konrad Lorenz**. Está diseñado para ayudar a estudiantes y docentes a consultar artículos de los reglamentos, entender sus derechos y obligaciones, y saber a qué instancia acudir según su situación académica.

El nombre **KonsultaBot** es un juego de palabras entre **Konrad** y **Consulta**.

---

## 🧠 Avance 2 — Pipeline RAG (`konsulta_bot_rag.py`)

### Descripción
Sistema de Retrieval-Augmented Generation (RAG) que indexa múltiples reglamentos en una base de datos vectorial y recupera solo los fragmentos más relevantes para cada consulta. A diferencia del Avance 1, no inyecta todo el documento en el prompt, solo los fragmentos necesarios.

### Flujo del sistema RAG

```
PDFs → CARGA (pypdf)
     → CHUNKING (RecursiveCharacterTextSplitter, 500 chars, overlap 50)
     → EMBEDDINGS (SentenceTransformers local — paraphrase-multilingual-MiniLM-L12-v2)
     → BASE VECTORIAL (ChromaDB, similitud coseno)
     → CONSULTA del usuario
     → RETRIEVAL (TOP-5 chunks más similares)
     → PROMPT AUMENTADO (system prompt + few-shot + contexto recuperado)
     → GEMINI 2.5 Flash
     → JSON estructurado
```

### Reglamentos indexados

| Reglamento | Páginas | Caracteres |
|-----------|---------|------------|
| Reglamento Académico de Pregrado | 52 | 109,733 |
| Reglamento Académico Institucional | 64 | 117,256 |
| Reglamento Docente | 35 | 62,060 |
| Reglamento Académico de Posgrado | 34 | 80,285 |
| **Total** | **185** | **369,334** |

**Total chunks indexados: 808**

### Componentes del pipeline

| Componente | Tecnología | Descripción |
|-----------|-----------|-------------|
| Extracción PDF | `pypdf` | Lee y extrae texto de los PDFs |
| Chunking | `langchain-text-splitters` | Divide el texto en fragmentos de 500 chars con overlap de 50 |
| Embeddings | `sentence-transformers` | Vectoriza los chunks localmente, sin API ni cuota |
| Base vectorial | `ChromaDB` | Almacena y consulta los embeddings por similitud coseno |
| LLM | `Google Gemini 2.5 Flash` | Genera la respuesta final en JSON |
| Interfaz | `colorama` | Interfaz de consola con colores |

### Configuración del sistema

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 500 | Tamaño de cada fragmento en caracteres |
| `CHUNK_OVERLAP` | 50 | Solapamiento entre fragmentos consecutivos |
| `TOP_K` | 5 | Número de chunks recuperados por consulta |
| `COLLECTION` | konsulta_reglamentos | Nombre de la colección en ChromaDB |
| `MODELO_ST` | paraphrase-multilingual-MiniLM-L12-v2 | Modelo de embeddings local |
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

**Few-Shot (3 ejemplos):**

| Pregunta | Artículo | Tipo |
|---------|---------|------|
| ¿Cuántas fallas me reprueban? | Art. 44 | Inasistencias |
| Perdí 4 materias, ¿qué pasa? | Art. 76 | Pérdida de cupo |
| ¿Cuál es el horario de la cafetería? | null | Fuera de dominio |

**Delimitador XML del contexto recuperado:**
```xml
<CONTEXTO_RAG>
  [fragmentos recuperados de ChromaDB]
</CONTEXTO_RAG>
```

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
│   ├── reglamento-académico-institucional.pdf
│   ├── reglamento-docente.pdf
│   └── reglamento-posgrado.pdf
├── .env.example
├── .gitignore
├── konsulta_bot_rag.py
├── README.md
└── requirements.txt
```

---

## 📦 Dependencias

```
# Avance 2 — RAG
chromadb>=0.5.0
sentence-transformers
langchain-text-splitters
colorama

# Avance 1 — Prompt Stuffing
google-genai>=1.0.0
pypdf>=4.0.0
python-dotenv>=1.0.0
```

---

## 🛠️ Tecnologías usadas

- **Python 3.10+**
- **Google Gemini 2.5 Flash** via `google-genai`
- **pypdf** — extracción de texto de PDFs
- **SentenceTransformers** — embeddings semánticos locales sin límites de API
- **ChromaDB** — base de datos vectorial persistente
- **LangChain Text Splitters** — chunking inteligente por separadores
- **colorama** — interfaz de consola con colores
- **Prompt Engineering**: System Prompt + Few-Shot + XML + JSON output

---

## 📎 Avance 1 — Prompt Stuffing (`Avance 1/konsulta_bot.py`)

> Versión inicial del proyecto. Documentada aquí como referencia del proceso.

### Descripción
El reglamento completo se extrae del PDF y se inyecta directamente en el prompt como contexto en cada consulta. El LLM responde con base en todo el texto del reglamento de pregrado.

### Técnicas de Prompting implementadas.

| Técnica | Descripción |
|---------|-------------|
| **System Prompt estructurado** | 6 secciones: ROL, TAREA, CONTEXTO, RESTRICCIONES, FORMATO, EJEMPLOS |
| **Few-Shot Prompting** | 5 ejemplos de pares pregunta/respuesta |
| **Delimitadores XML** | Etiquetas `<restriccion_*>` para separar comportamientos |
| **Formato JSON** | Salida estructurada con 5 campos fijos |
| **Prompt Stuffing** | Texto completo del PDF inyectado como contexto |

### Progreso entre Avance 1 y Avance 2

| | Avance 1 | Avance 2 |
|--|---------|---------|
| Contexto | Todo el reglamento en el prompt | Solo los 5 chunks más relevantes |
| Documentos | 1 PDF | 4 PDFs |
| Embeddings | No aplica | SentenceTransformers (local, sin cuota) |
| Base vectorial | No aplica | ChromaDB persistente |
| Escalabilidad | Limitada por tokens | Alta — soporta cualquier cantidad de PDFs |