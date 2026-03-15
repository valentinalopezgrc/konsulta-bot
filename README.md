# 🎓 KonsultaBot — Asistente del Reglamento Académico
### Fundación Universitaria Konrad Lorenz

> Proyecto: **Desarrollo de un Asistente Experto basado en RAG y Agentes**  

---

## ¿Qué es KonsultaBot?

**KonsultaBot** es un asistente conversacional experto en el **Reglamento Académico de Pregrado de la Fundación Universitaria Konrad Lorenz** (Resolución de Rectoría No. 01 — 19 de febrero de 2021).

Está diseñado para ayudar a **estudiantes, docentes y personal administrativo** a consultar artículos del reglamento, entender sus derechos y obligaciones, y saber exactamente a qué instancia acudir según su situación académica.

El nombre **KonsultaBot** es un juego de palabras entre **Konrad** (la universidad) y **Consulta** (su función principal).

---

## 🎯 Enfoque elegido

**Asistente Legal/Normativo** — basado en el Reglamento Académico de Pregrado oficial de la Fundación Universitaria Konrad Lorenz.

---

## 🧠 Técnicas de Prompting implementadas

### 1. System Prompt estructurado
El prompt está organizado en 6 secciones claramente definidas:

| Sección | Descripción |
|---------|-------------|
| **ROL** | Define quién es KonsultaBot y qué puede hacer |
| **TAREA** | Qué tipo de consultas resuelve y cómo |
| **CONTEXTO** | El texto completo del reglamento, delimitado con triple comilla `'''` |
| **RESTRICCIONES** | Comportamiento ante diferentes tipos de preguntas |
| **FORMATO** | Estructura JSON obligatoria para cada respuesta |
| **EJEMPLOS** | 5 casos de referencia (Few-Shot) |

### 2. Few-Shot Prompting
Se incluyen **5 ejemplos** de pares pregunta/respuesta que cubren los casos más frecuentes:

| Ejemplo | Pregunta | Artículo |
|---------|----------|----------|
| 1 | ¿Cuántas fallas puedo tener? | Art. 44, Párr. 1 |
| 2 | Perdí 4 materias, ¿qué pasa? | Art. 76 |
| 3 | ¿Cuánto tiempo tengo para revisar una nota? | Art. 72 |
| 4 | ¿Me pueden expulsar por hacer trampa? | Arts. 107 + 108 |
| 5 | ¿Qué pasa si no me matriculo a tiempo? | Art. 42 |

### 3. Delimitadores XML
Cada restricción de comportamiento está envuelta en etiquetas XML para separar claramente el contexto de las instrucciones:

```xml
<restriccion_fuera_dominio>   → preguntas no relacionadas con el reglamento
<restriccion_ambigua>         → preguntas incompletas o sin contexto
<restriccion_disciplinaria>   → situaciones del régimen disciplinario (Cap. XV)
<restriccion_calificaciones>  → consultas sobre notas y escalas
<restriccion_perdida_cupo>    → situaciones críticas académicas
```

### 4. Formato de salida estructurado (JSON)
Cada respuesta devuelve siempre un JSON con 5 campos fijos:

```json
{
  "articulo": "Art. XX [+ Parágrafo si aplica]",
  "respuesta": "Explicación clara y accesible",
  "cita_textual": "Fragmento literal del reglamento",
  "accion_recomendada": "Qué hacer / a quién acudir",
  "advertencia": "Riesgo académico o disciplinario, o null"
}
```

### 5. Reglamento inyectado desde el PDF
El texto real del reglamento se extrae del PDF en tiempo de ejecución con `pypdf` y se inyecta directamente en el prompt como contexto. Esto garantiza que KonsultaBot cite artículos reales y no inventados.

```python
# El reglamento se carga así:
texto_reglamento = cargar_reglamento("reglamento-academico-de-pregrado.pdf")
system_prompt = construir_system_prompt(texto_reglamento)
```

---

## 🗂️ Estructura del repositorio

```
├── konsulta_bot.py                        # Código fuente principal
├── .env.example                           # Plantilla de variables de entorno
├── reglamento-academico-de-pregrado.pdf   # Documento base oficial
├── .gitignore                             # Excluye .env y archivos innecesarios
├── requirements.txt                       # Dependencias del proyecto
└── README.md                              # Resumen del proyecto
```
⚠️ NOTA IMPORTANTE: La API key debe generarse desde un proyecto nuevo en 
Google AI Studio. El free tier tiene límite de tokens por minuto,
por lo que se recomienda generar desde UN PROYECTO NUEVO en 
Google AI Studio (aistudio.google.com/apikey). Si el proyecto 
ya tiene cuota agotada, el bot devolverá error 429.
---

## ⚙️ Instalación y ejecución

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
# Editar .env y agregar: GENAI_API_KEY=tu_clave_aqui

# 5. Ejecutar
python konsulta_bot.py
```

---

## 💬 Ejemplo de ejecución

```
Cargando reglamento desde PDF...
Reglamento cargado: 87,432 caracteres extraídos.

================================================================
🎓 KONSULTA BOT — ASISTENTE DEL REGLAMENTO ACADÉMICO
Fundación Universitaria Konrad Lorenz
================================================================

TÚ: ¿Cuántas fallas puedo tener antes de perder una materia?

🎓 KONSULTA BOT:
{
  "articulo": "Art. 44, Parágrafo 1",
  "respuesta": "Puedes acumular máximo un 20% de inasistencias...",
  "cita_textual": "El número máximo de fallas que un estudiante...",
  "accion_recomendada": "Lleva un conteo personal de tus asistencias...",
  "advertencia": null
}
```

---

## 📦 Dependencias

```
google-genai
python-dotenv
pypdf
```

---

## 🛠️ Tecnologías usadas

- **Python 3.10+**
- **Google Gemini 2.5 Flash** via `google-genai`
- **pypdf** — extracción de texto del PDF
- **Prompt Engineering**: System Prompt estructurado + Few-Shot + Delimitadores XML + JSON output
