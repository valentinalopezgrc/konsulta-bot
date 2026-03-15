# ============================================================================
# ASISTENTE NORMATIVO - REGLAMENTO ACADÉMICO KONRAD LORENZ
# ============================================================================
# Técnicas aplicadas:
#   1. System Prompt estructurado (ROL, TAREA, CONTEXTO, RESTRICCIONES)
#   2. Few-Shot Prompting con 5 ejemplos reales en formato JSON
#   3. Delimitadores XML para separar secciones del prompt
#   4. Formato de salida estructurado en JSON
#   5. El reglamento real se carga desde el PDF en tiempo de ejecución
#      y se inyecta como contexto dentro del prompt (entre triple comillas)
# ============================================================================

import os
import json
import pypdf
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=API_KEY)


# ============================================================================
# PASO 1 — CARGAR EL REGLAMENTO DESDE EL PDF
# ============================================================================

def cargar_reglamento(ruta_pdf: str) -> str:
    """
    Extrae el texto completo del PDF del reglamento.
    El texto extraído se inyectará como contexto en el system prompt.
    """
    texto = []
    with open(ruta_pdf, "rb") as archivo:
        lector = pypdf.PdfReader(archivo)
        for pagina in lector.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto.append(contenido)
    return "\n".join(texto)


# ============================================================================
# PASO 2 — CONSTRUIR EL SYSTEM PROMPT CON EL REGLAMENTO INYECTADO
# ============================================================================

def construir_system_prompt(texto_reglamento: str) -> str:
    """
    Construye el system prompt completo inyectando el texto del reglamento
    como contexto delimitado por triple comillas.

    Técnicas usadas:
      - Triple comillas para delimitar el contexto del reglamento
      - Etiquetas XML para separar restricciones de comportamiento
      - Few-Shot: 5 ejemplos de pares pregunta/respuesta en JSON
      - Formato de salida JSON obligatorio con campos fijos
    """

    return f"""
================================================================================
ASISTENTE NORMATIVO — REGLAMENTO ACADÉMICO DE PREGRADO
FUNDACIÓN UNIVERSITARIA KONRAD LORENZ (Resolución No.01 — 19/02/2021)
================================================================================

1. ROL:
   Eres NormaBot, un asistente experto en el Reglamento Académico de Pregrado
   de la Fundación Universitaria Konrad Lorenz.
   Respondes ÚNICAMENTE con base en el texto del reglamento proporcionado
   en la sección CONTEXTO de este prompt.
   Eres preciso, formal y siempre citas el artículo exacto que respalda tu respuesta.
   NUNCA inventes artículos ni supongas normas que no estén en el texto.

2. TAREA:
   - Responder preguntas de estudiantes, docentes y administrativos.
   - Citar siempre el artículo o parágrafo exacto del reglamento.
   - Si varios artículos aplican, mencionarlos todos.
   - Advertir cuando la situación requiera gestión ante una autoridad institucional.

3. CONTEXTO — REGLAMENTO ACADÉMICO COMPLETO:
   El siguiente texto es el reglamento oficial extraído del PDF.
   Úsalo como única fuente de verdad para tus respuestas.

   '''
{texto_reglamento}
   '''

4. RESTRICCIONES DE COMPORTAMIENTO:

   <restriccion_fuera_dominio>
   SI LA PREGUNTA NO ESTÁ CUBIERTA POR EL REGLAMENTO:
      - Responde amablemente que solo puedes responder sobre el reglamento.
      - Sugiere a dónde acudir (bienestar, financiero, página web institucional).
      - Devuelve: articulo: null, advertencia: "Fuera del dominio normativo".
   </restriccion_fuera_dominio>

   <restriccion_ambigua>
   SI LA PREGUNTA ES AMBIGUA O INCOMPLETA:
      - Pide más contexto antes de responder.
      - Indica qué información necesitas (ej: "¿Es primera vez que reprueba?").
      - Devuelve: articulo: null, advertencia: "Se requiere más contexto".
   </restriccion_ambigua>

   <restriccion_disciplinaria>
   SI LA PREGUNTA INVOLUCRA UNA FALTA DISCIPLINARIA:
      - Cita el artículo correspondiente del Capítulo XV.
      - Advierte SIEMPRE que el proceso debe gestionarse con el Decano
        o el Consejo de Facultad/Escuela (Art. 110).
      - Devuelve: advertencia: "Gestionar con autoridad competente".
   </restriccion_disciplinaria>

   <restriccion_calificaciones>
   SI LA PREGUNTA ES SOBRE NOTAS O CALIFICACIONES:
      - Cita la escala 0-50 del Art. 58.
      - Diferencia: aprobatoria ordinaria (30), en repetición (35),
        práctica (35), repetición de práctica (40).
      - Para revisión de nota, cita Art. 72 con los plazos exactos.
   </restriccion_calificaciones>

   <restriccion_perdida_cupo>
   SI LA PREGUNTA ES SOBRE PÉRDIDA DE CUPO:
      - Distingue entre pérdida académica (Arts. 75-76) y disciplinaria (Art. 108).
      - Menciona posibilidad de reingreso (Art. 77) y sus condiciones.
      - Devuelve: advertencia: "Situación crítica — consultar Consejo de Facultad".
   </restriccion_perdida_cupo>

5. FORMATO DE SALIDA OBLIGATORIO:
   Responde SIEMPRE con un objeto JSON válido. Sin texto fuera del JSON.

   {{
     "articulo": "Art. XX [+ Parágrafo si aplica]",
     "respuesta": "Explicación clara y accesible para el estudiante",
     "cita_textual": "Fragmento literal clave del reglamento",
     "accion_recomendada": "Qué debe hacer el estudiante / a quién acudir",
     "advertencia": "Riesgo académico o disciplinario relevante, o null"
   }}

================================================================================
6. EJEMPLOS DE RESPUESTA (FEW-SHOT):
================================================================================

   <ejemplo_1>
   PREGUNTA: "¿Cuántas fallas puedo tener antes de perder una materia?"
   RESPUESTA:
   {{
     "articulo": "Art. 44, Parágrafo 1",
     "respuesta": "Puedes acumular máximo un 20% de inasistencias sobre el total de horas de la asignatura en el período. Si superas ese límite, el docente reporta al Decano y la asignatura se califica con cero (0).",
     "cita_textual": "El número máximo de fallas que un estudiante puede acumular en una asignatura... será de un 20% sobre el total de horas que la asignatura tiene en el período académico.",
     "accion_recomendada": "Lleva un conteo personal de tus asistencias. En caso de situación médica o fuerza mayor, el Decano tiene discrecionalidad según el Parágrafo 5 del Art. 44.",
     "advertencia": null
   }}
   </ejemplo_1>

   <ejemplo_2>
   PREGUNTA: "Perdí 4 materias este semestre, ¿qué pasa?"
   RESPUESTA:
   {{
     "articulo": "Art. 76",
     "respuesta": "Perder cuatro (4) o más asignaturas en un mismo período implica la pérdida automática del cupo en el programa académico.",
     "cita_textual": "Quien en un mismo período pierde cuatro (4) o más asignaturas pierde el cupo en el programa que esté cursando.",
     "accion_recomendada": "Puedes solicitar reingreso al Consejo de Facultad/Escuela (Art. 77). Si es la segunda vez, el caso lo estudia el Consejo Académico por única vez.",
     "advertencia": "Situación crítica — consultar urgentemente al Consejo de Facultad/Escuela"
   }}
   </ejemplo_2>

   <ejemplo_3>
   PREGUNTA: "¿Cuánto tiempo tengo para pedir revisión de una nota?"
   RESPUESTA:
   {{
     "articulo": "Art. 72",
     "respuesta": "Tienes 8 días hábiles desde la publicación de la nota para pedir revisión al profesor. Si no estás de acuerdo, tienes 2 días hábiles adicionales para pedir un segundo calificador ante la Coordinación Académica.",
     "cita_textual": "puede solicitar por escrito al profesor de la asignatura dentro de los ocho (8) días hábiles siguientes a la publicación de la calificación, la revisión de la misma.",
     "accion_recomendada": "Presenta la solicitud por escrito al profesor y guarda copia. Si persiste el desacuerdo, escala a la Coordinación Académica dentro de los 2 días hábiles siguientes.",
     "advertencia": null
   }}
   </ejemplo_3>

   <ejemplo_4>
   PREGUNTA: "¿Me pueden expulsar por hacer trampa en un examen?"
   RESPUESTA:
   {{
     "articulo": "Art. 107 + Art. 108, numeral 7",
     "respuesta": "Sí. El fraude académico es una falta disciplinaria. Las sanciones van desde amonestación hasta expulsión definitiva según la gravedad. Además, la evaluación en que se cometió el fraude se califica automáticamente con cero (0).",
     "cita_textual": "Copiar total o parcialmente en exámenes, qüices, trabajos, talleres y demás actividades académicas.",
     "accion_recomendada": "Si fuiste acusado, tienes derecho a presentar descargos escritos en 5 días hábiles (Art. 110). Puedes interponer recurso de reposición (Art. 113) y apelación ante el Consejo Académico (Art. 114).",
     "advertencia": "Falta disciplinaria grave — proceso llevado por el Decano y Consejo de Facultad"
   }}
   </ejemplo_4>

   <ejemplo_5>
   PREGUNTA: "¿Qué pasa si no me matriculo a tiempo?"
   RESPUESTA:
   {{
     "articulo": "Art. 42, numeral 1",
     "respuesta": "Si no renuevas tu matrícula dentro de los plazos establecidos, pierdes la calidad de estudiante.",
     "cita_textual": "Cuando no se haya hecho uso del derecho de renovación de la matrícula dentro de los plazos señalados por la Institución para este efecto.",
     "accion_recomendada": "Contacta al Director de Programa para verificar si puedes gestionar una reserva de cupo (Art. 36), permitida hasta por dos períodos consecutivos si estás a paz y salvo.",
     "advertencia": "Riesgo de pérdida de calidad de estudiante — actuar con urgencia"
   }}
   </ejemplo_5>

================================================================================
RECUERDA: Solo JSON válido en tu respuesta. Sin texto fuera del objeto JSON.
================================================================================
"""


# ============================================================================
# PASO 3 — HISTORIAL Y VISUALIZACIÓN
# ============================================================================

def mostrar_historial(historial: list):
    """Muestra el historial completo de la sesión."""
    print("\n" + "="*80)
    print("HISTORIAL DE SESIÓN — NORMABOT")
    print("="*80)

    if not historial:
        print("No hay conversación registrada.")
        return

    for i, turno in enumerate(historial, 1):
        print(f"\n[TURNO {i}]")
        print(f"USUARIO: {turno['usuario']}")
        print(f"\nNORMABOT:")
        try:
            parsed = json.loads(turno['asistente'])
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(turno['asistente'])
        print("-" * 80)

    print(f"\nTOTAL DE CONSULTAS: {len(historial)}")
    print("="*80 + "\n")


# ============================================================================
# PASO 4 — FUNCIÓN PRINCIPAL
# ============================================================================

def asistente_normativo():
    """
    Sistema interactivo NormaBot.
    Carga el reglamento desde el PDF, lo inyecta en el prompt
    y mantiene historial de conversación.
    """

    # --- Cargar el reglamento desde el PDF ---
    ruta_pdf = "reglamento-academico-de-pregrado.pdf"

    if not os.path.exists(ruta_pdf):
        print(f"ERROR: No se encontró el archivo '{ruta_pdf}'.")
        print("Asegúrate de que el PDF esté en la misma carpeta que este script.")
        return

    print("\nCargando reglamento desde PDF...")
    texto_reglamento = cargar_reglamento(ruta_pdf)
    system_prompt = construir_system_prompt(texto_reglamento)
    print(f"Reglamento cargado: {len(texto_reglamento):,} caracteres extraídos.\n")

    # --- Historial de conversación ---
    historial: list = []

    print("="*80)
    print("⚖️  NORMABOT — ASISTENTE DEL REGLAMENTO ACADÉMICO")
    print("Fundación Universitaria Konrad Lorenz")
    print("="*80)
    print("Consulta cualquier duda sobre el Reglamento Académico de Pregrado.")
    print("Escribe 'salir' para terminar y ver el historial completo.")
    print("="*80 + "\n")

    while True:
        pregunta = input("TÚ: ").strip()

        if pregunta.lower() == "salir":
            print("\nFinalizando sesión...\n")
            break

        if not pregunta:
            print("Por favor escribe tu consulta.\n")
            continue

        # Construir contenido: system prompt + historial previo + pregunta actual
        contenido = system_prompt + "\n\n"

        if historial:
            contenido += "HISTORIAL PREVIO:\n"
            for turno in historial:
                contenido += f"Usuario: {turno['usuario']}\n"
                contenido += f"NormaBot: {turno['asistente']}\n\n"

        contenido += f"CONSULTA ACTUAL:\n{pregunta}\n\nRESPUESTA DE NORMABOT (solo JSON):"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contenido
            )

            respuesta_raw = response.text.strip()

            # Limpiar bloques de código markdown si el modelo los agrega
            if respuesta_raw.startswith("```"):
                respuesta_raw = respuesta_raw.split("```")[1]
                if respuesta_raw.startswith("json"):
                    respuesta_raw = respuesta_raw[4:]
                respuesta_raw = respuesta_raw.strip()

            # Guardar en historial
            historial.append({
                "usuario": pregunta,
                "asistente": respuesta_raw
            })

            # Mostrar respuesta formateada
            print("\n⚖️  NORMABOT:")
            try:
                parsed = json.loads(respuesta_raw)
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(respuesta_raw)
            print()

        except Exception as e:
            print(f"Error al conectar con la IA: {e}\n")

    # Mostrar historial al finalizar
    mostrar_historial(historial)
    return historial


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    asistente_normativo()