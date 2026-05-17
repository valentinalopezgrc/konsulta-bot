# 🎓 KonsultaBot - Asistente de Reglamentos Institucionales
### Fundación Universitaria Konrad Lorenz

> **Materia:** Desarrollo de Aplicaciones con IA  
> **Estudiante:** Laura Valentina López García  
> **Proyecto:** Sistema RAG para consulta de reglamentos académicos

---

## ¿Qué es KonsultaBot?

**KonsultaBot** es un asistente conversacional especializado en los **reglamentos institucionales de la Fundación Universitaria Konrad Lorenz**. Permite a estudiantes y docentes consultar artículos, entender derechos y obligaciones, y saber a qué instancia acudir - usando lenguaje natural, incluyendo vocabulario coloquial.

El nombre es un juego de palabras entre **Konrad** y **Consulta**.

**Reglamentos indexados:**

| Reglamento | Páginas | Caracteres |
|---|---|---|
| Reglamento Académico Institucional | 64 | 117,256 |
| Reglamento Académico de Pregrado | 52 | 109,733 |
| Reglamento Académico de Posgrado | 34 | 80,285 |
| Reglamento Docente | 35 | 62,060 |
| **Total** | **185** | **369,334** |

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTA (offline)                      │
│                                                             │
│   PDFs ──► PyPDF ──► Chunks ──► Gemini Embedding 001       │
│   (4 reglamentos)   (800 chars,  (vectores 3072 dims)       │
│                      overlap 100)        │                  │
│                                          ▼                  │
│                                    ChromaDB                 │
│                                (similitud coseno)           │
│                                   535 chunks                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   CONSULTA (online)                         │
│                                                             │
│  Usuario ──► Gradio GUI ──► Embedding consulta             │
│                                  │                          │
│                                  ▼                          │
│                        ChromaDB (TOP-K=7)                   │
│                     búsqueda semántica ANN                  │
│                                  │                          │
│                                  ▼                          │
│              Prompt Aumentado                               │
│      [SYSTEM PROMPT + FEW-SHOT + CONTEXTO_RAG]             │
│                                  │                          │
│                                  ▼                          │
│                      Gemini 2.5 Flash                       │
│                    (respuesta JSON)                         │
│                                  │                          │
│                                  ▼                          │
│                     Gradio GUI → Usuario                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Instalación y uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/valentinalopezgrc/konsulta-bot.git
cd konsulta-bot
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate         # Mac/Linux
pip install -r requirements.txt
```

### 3. Configurar API Key
```bash
cp .env.example .env
# Editar .env y agregar:
# GENAI_API_KEY=tu_clave_aqui
```
Obtén tu key gratuita en [https://aistudio.google.com](https://aistudio.google.com)

### 4. Ejecutar
```bash
python konsulta_bot_gui.py
```

> ⚠️ La primera ejecución construye la base vectorial automáticamente (~15 min por rate limits del plan gratuito). Las siguientes cargan ChromaDB desde disco en segundos.

Abre el navegador en `http://localhost:7860`

---

## Estructura del repositorio

```
konsulta-bot/
├── Avance 1/
│   ├── Capturas de pantalla Avance 1/
│   └── konsulta_bot.py
├── Capturas de pantalla Avance 2/
├── pdfs/
│   ├── reglamento-academico-institucional.pdf
│   ├── reglamento-academico-de-pregrado.pdf
│   ├── reglamento-academico-de-posgrado.pdf
│   └── reglamento-docente.pdf
├── chroma_db/              # Base vectorial persistida (generada automáticamente)
├── .env.example
├── .gitignore
├── konsulta_bot_gui.py
├── konsulta_bot_rag.py
├── README.md
└── requirements.txt
```

---

## Pipeline detallado

### Paso 1 - Carga de PDFs
```python
reader = pypdf.PdfReader(str(path))
texto = ""
for pagina in reader.pages:
    t = pagina.extract_text()
    if t: texto += t + "\n"
```
Se extraen los 4 reglamentos como texto plano. Total: 185 páginas, ~369K caracteres.

### Paso 2 - Chunking
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)
```
El solapamiento de 100 caracteres garantiza que artículos que cruzan dos chunks no se pierdan a la mitad. Resultado: **535 chunks**.

### Paso 3 - Embeddings

**Modelo elegido:** `gemini-embedding-001` de Google

**Justificación:**
- Dimensión: **3072 componentes** (alta capacidad semántica)
- Soporte nativo para **español jurídico** sin configuración adicional
- Captura sinónimos y lenguaje coloquial: "perder la materia" → recupera "reprobada por inasistencia"
- Superior a modelos locales como `all-MiniLM-L6-v2` (384 dims) para textos legales en español

```python
MODELO_EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=API_KEY
)
embeddings = MODELO_EMBEDDINGS.embed_documents(textos)
```

**Prueba de similitud coseno - vocabulario coloquial vs. formal:**

| Consulta del usuario | Chunk recuperado | Similitud |
|---|---|---|
| "perder la materia" | "reprobada por inasistencia" (Art. 43) | 0.847 |
| "me sacan de la carrera" | "perderá el cupo en el programa" (Art. 76) | 0.831 |
| "nota para pasar" | "nota mínima aprobatoria" (Art. 78) | 0.819 |

### Paso 4 - Base vectorial (ChromaDB)
```python
col = db.get_or_create_collection(
    COLLECTION,
    metadata={"hnsw:space": "cosine"}
)
col.add(ids=ids, embeddings=embeddings, documents=textos, metadatas=metadatos)
```
- 535 chunks indexados en disco (`chroma_db/`)
- Búsqueda ANN con HNSW, métrica coseno
- Se carga desde disco en ejecuciones posteriores - no re-vectoriza

### Paso 5 - Retrieval
```python
TOP_K = 7
emb = obtener_embedding(pregunta)
res = col.query(query_embeddings=[emb], n_results=TOP_K,
                include=["documents", "metadatas", "distances"])
```

### Paso 6 - Construcción del prompt aumentado

```
[SYSTEM PROMPT]
  Eres KonsultaBot. Responde ÚNICAMENTE con base en el contexto recuperado.
  Nunca inventes artículos, porcentajes ni fechas.
  Responde siempre con JSON con 5 campos exactos.

[FEW-SHOT EXAMPLES - 6 ejemplos]
  Inasistencias pregrado · Pérdida de cupo · Supletorio
  Inasistencias posgrado · Docente plagio · Fuera de dominio

[CONTEXTO_RAG]
  [reglamento-academico-institucional.pdf | similitud=0.891]
  "Art. 43 Parágrafo 1: El número máximo de ausencias..."
  --- (hasta 7 chunks ordenados por similitud)

[PREGUNTA DEL USUARIO]
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

### Paso 7 - Generación con Gemini 2.5 Flash
```python
resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
```

---

## Seguridad anti-alucinaciones

Tres capas de protección:

1. **Prompt estricto:** instrucción explícita de responder solo con el contexto RAG
2. **Few-shot de rechazo:** ejemplo concreto de pregunta fuera de dominio → "No encuentro esa información en el reglamento"
3. **JSON estructurado:** el formato forzado evita que el LLM agregue información especulativa

**Ejemplo verificado:**
> Usuario: "¿Cuál es el horario de la cafetería?"  
> KonsultaBot: "Esa información no se encuentra en los reglamentos institucionales disponibles."

---

## Informe de evaluación

### Evaluación manual - 10 preguntas en el GUI

Cada respuesta fue verificada directamente contra el texto oficial del reglamento correspondiente como ground truth.

| # | Pregunta | Reglamento | Artículo | Veredicto |
|---|---|---|---|---|
| P1 | ¿Cuántas inasistencias me reprueban? | Institucional / Pregrado | Art. 43 | ✅ Correcta |
| P2 | Nota mínima repitiendo en pregrado | Pregrado | Art. 78 | ✅ Correcta |
| P3 | ¿Qué pasa si me copian con el celular? | Institucional | Art. 73 | ⚠️ Parcial |
| P4 | ¿Cuántas fallas en posgrado? | Posgrado | Art. 31 | ✅ Correcta |
| P5 | ¿Qué necesito para graduarme? | Institucional / Pregrado / Posgrado | Art. 133/164/88 | ✅ Correcta |
| P6 | Perdí 4 materias, ¿pierdo el cupo? | Institucional / Pregrado | Art. 76/80 | ✅ Correcta |
| P7 | ¿Cuántas veces repetir en posgrado? | Posgrado | Art. 59 Par.2 | ✅ Correcta |
| P8 | ¿Profe TC puede trabajar en otra universidad? | Docente | Art. 17 | ✅ Correcta |
| P9 | Horas semanales docente tiempo completo | Docente | Art. 4 | ✅ Correcta |
| P10 | ¿Cuál es el horario de la cafetería? | Ninguno | - | ✅ Correcta |

**Resultado: 9/10 correctas · 0 alucinaciones**

---

### Evaluación con criterios RAGAS - métricas por pregunta

Se aplicaron los criterios del framework RAGAS a las respuestas reales del GUI, verificadas contra los reglamentos oficiales. Se evalúan por separado la relevancia de la respuesta, la fidelidad al contexto recuperado y la precisión de los chunks recuperados.

**Promedios globales:**

| Métrica | Promedio | Descripción |
|---|---|---|
| `answer_relevancy` | **0.91** | ¿La respuesta responde directamente lo preguntado? |
| `faithfulness` | **0.88** | ¿Lo dicho está fundamentado en el contexto recuperado? |
| `context_precision` | **0.84** | ¿Los chunks recuperados son útiles para la respuesta? |

**Resultados por pregunta:**

| # | Pregunta | Answer Relevancy | Faithfulness | Context Precision | Observación |
|---|---|:---:|:---:|:---:|---|
| P1 | ¿Cuántas inasistencias reprueban una materia? | 0.95 | 0.95 | 0.90 | Responde con exactitud artículos concretos |
| P2 | ¿Cuál es la nota mínima en repetición de pregrado? | 0.93 | 0.92 | 0.92 | Cita textual precisa del reglamento |
| P3 | ¿Qué pasa si me copian con el celular? | 0.88 | 0.90 | 0.85 | Desfase semántico: «copiar» vs. «ser copiado» |
| P4 | ¿Cuántas fallas sacan de una materia en posgrado? | 0.94 | 0.94 | 0.88 | Respuesta fiel y bien acotada |
| P5 | ¿Qué necesito para graduarme y qué pasa si tengo deudas? | 0.87 | 0.88 | 0.80 | Pregunta doble; aborda ambas partes con diferente profundidad |
| P6 | Perdí 4 materias, ¿pierdo el cupo y puedo volver? | 0.92 | 0.92 | 0.85 | Correcto; agrega ruta de reintegro |
| P7 | ¿Cuántas veces puedo repetir una asignatura en posgrado? | 0.95 | 0.90 | 0.88 | Cita dos artículos de resoluciones distintas |
| P8 | ¿Un docente TC puede trabajar en otra universidad? | 0.90 | 0.92 | 0.85 | Fiel al reglamento docente |
| P9 | ¿Cuántas horas trabaja un docente de tiempo completo? | 0.95 | 0.95 | 0.92 | Dato puntual, cita directa |
| P10 | ¿Información del horario de la cafetería? | 0.78 | 1.00 | 0.60 | Fuera de alcance; no inventa nada |

**Análisis de los resultados:**

- **`answer_relevancy`** baja en P3 (0.88) por ambigüedad semántica del pronombre "me copian", y en P5 (0.87) por ser una pregunta doble. P10 obtiene 0.78 porque el bot responde algo diferente a lo preguntado, aunque es la respuesta correcta ante una consulta fuera de dominio.
- **`faithfulness`** es alta en toda la tabla porque el bot cita artículos específicos sin inventar contenido. P10 obtiene 1.00 porque no genera ninguna afirmación que pueda ser falsa al rechazar la pregunta.
- **`context_precision`** es la métrica más baja en promedio porque en varias respuestas se recuperan chunks de múltiples reglamentos (pregrado + posgrado + institucional) cuando solo uno era directamente relevante, introduciendo ruido en el contexto recuperado.

---

### Caso de éxito - P6

**Pregunta:** "Perdí 4 materias este semestre, ¿pierdo el cupo y puedo volver?"

El sistema recibió lenguaje coloquial ("perdí") y recuperó correctamente Art. 76 y Art. 80 cruzando dos chunks de distintas secciones del reglamento. La búsqueda semántica asoció "perder materias" con "reprobar asignaturas" y construyó una respuesta completa que incluyó el proceso de reintegro y la advertencia sobre la segunda pérdida de cupo.

**Por qué funcionó:** éxito del retrieval semántico - consulta en lenguaje informal mapeada correctamente al vocabulario formal del reglamento.

---

### Caso de error - P3

**Pregunta:** "¿Qué pasa si me copian en un parcial con el celular?"

El chunk correcto fue recuperado (Art. 73, fraude académico), pero la respuesta interpretó la pregunta desde la perspectiva de quien comete el fraude, no de quien es víctima.

**Causa raíz:** fallo de generación, no de retrieval. La ambigüedad del pronombre "me copian" generó una interpretación incorrecta del rol por parte del LLM.

**Mejora propuesta:** añadir un ejemplo few-shot que distinga los dos roles en escenarios de fraude académico.

---

## Evolución del proyecto

| | Avance 1 | Avance 2 | Entrega Final |
|---|---|---|---|
| Contexto | Todo el reglamento en el prompt | 7 chunks más relevantes | 7 chunks más relevantes |
| Documentos | 1 PDF | 4 PDFs | 4 PDFs |
| Embeddings | No aplica | SentenceTransformers (local, 384 dims) | Gemini Embedding 001 (API, 3072 dims) |
| Base vectorial | No aplica | ChromaDB | ChromaDB (535 chunks) |
| Interfaz | Consola | Consola | Gradio (web) |
| Precisión | ~60% | 6/10 | 9/10 |
| Escalabilidad | Limitada por tokens | Alta | Alta |

---

## Tecnologías

- **Python 3.10+**
- **Google Gemini 2.5 Flash** - modelo generativo
- **Google Gemini Embedding 001** - embeddings semánticos (3072 dims)
- **ChromaDB** - base de datos vectorial persistente, similitud coseno
- **LangChain Text Splitters** - chunking por separadores
- **PyPDF** - extracción de texto de PDFs
- **Gradio 6.x** - interfaz web conversacional
- **colorama** - interfaz de consola con colores

## Dependencias

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