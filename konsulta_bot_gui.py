"""
KonsultaBot GUI — Interfaz Gradio
Estudiante: Laura Valentina López García
Materia: Desarrollo de Aplicaciones con IA — Fundación Universitaria Konrad Lorenz
"""

import gradio as gr
from gradio import ChatMessage
from pathlib import Path
from konsulta_bot_rag import (
    cargar_pdfs, crear_chunks, construir_base_vectorial,
    cargar_base_vectorial_existente, recuperar_chunks, generar_respuesta,
    PDF_DIR, CHROMA_DIR
)

# ══════════════════════════════════════════
# INICIALIZAR RAG
# ══════════════════════════════════════════
print("Iniciando KonsultaBot...")
chroma_existe = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
if chroma_existe:
    col = cargar_base_vectorial_existente()
else:
    docs   = cargar_pdfs(PDF_DIR)
    chunks = crear_chunks(docs)
    col    = construir_base_vectorial(chunks)
print("Sistema RAG listo.")

BIENVENIDA = "Hola 👋 Soy **KonsultaBot**. Puedo resolver tus dudas sobre los reglamentos de la Fundación Universitaria Konrad Lorenz. ¿En qué te puedo ayudar hoy?"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --kl-magenta: #E5007D;
    --kl-lima:    #97C11F;
    --kl-teal:    #5B9EA0;
    --kl-dark:    #1a1a2e;
    --kl-bg:      #0f0f1a;
    --body-background-fill: #16213e;
    --button-primary-background-fill: #E5007D;
    --button-primary-background-fill-hover: #bf0069;
    --button-primary-border-color: #E5007D;
}

.dark {
    --body-background-fill: #16213e !important;
    --button-primary-background-fill: #E5007D !important;
    --button-primary-background-fill-hover: #bf0069 !important;
    --button-primary-border-color: #E5007D !important;
}

/* ── Base ── */
html, body, .gradio-container, #root, .main, .wrap {
    background: #16213e !important;
    font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
}
gradio-app {
    background: #16213e !important;
    --body-background-fill: #16213e !important;
}
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
}
footer { display: none !important; }

/* ── Tipografía general ── */
.gradio-container h1 {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}
.gradio-container h3 {
    color: #ffffff !important;
    font-weight: 400 !important;
    font-size: 14px !important;
}
.gradio-container p {
    color: #ffffff !important;
}

/* ── Chatbot container ── */
#chatbot {
    background: #16213e !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 12px !important;
}

/* ── Ocultar label ── */
.label-wrap { display: none !important; }

/* ── Ocultar botones share/copy/delete ── */
.message-buttons,
.copy-btn,
button.share-button,
[aria-label="Delete message"],
[aria-label="Copy message"] {
    display: none !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Burbuja usuario ── */
#chatbot [data-testid="user"],
#chatbot .user {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot [data-testid="user"] > div:first-child,
#chatbot .user > div:first-child {
    background: var(--kl-magenta) !important;
    color: #fff !important;
    border-radius: 16px 4px 16px 16px !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 12px 16px !important;
}

/* ── Burbuja bot ── */
#chatbot [data-testid="bot"],
#chatbot .bot,
.bot.svelte-1nr59td,
div.bot.svelte-1nr59td {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#chatbot [data-testid="bot"] > div:first-child,
#chatbot .bot > div:first-child,
.bot.svelte-1nr59td > div:first-child {
    background: #1e2a4a !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 4px 16px 16px 16px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 12px 16px !important;
}

/* ── Markdown dentro del bot ── */
#chatbot [data-testid="bot"] strong,
#chatbot .bot strong {
    color: var(--kl-lima) !important;
}
#chatbot [data-testid="bot"] blockquote,
#chatbot .bot blockquote {
    background: rgba(151, 193, 31, 0.08) !important;
    border-left: 3px solid var(--kl-lima) !important;
    border-radius: 0 6px 6px 0 !important;
    margin: 8px 0 !important;
    padding: 8px 12px !important;
    color: #b0c98a !important;
    font-style: italic !important;
}
#chatbot [data-testid="bot"] hr,
#chatbot .bot hr {
    border-color: #2d3d6e !important;
    margin: 10px 0 8px !important;
}
#chatbot [data-testid="bot"] code,
#chatbot .bot code {
    background: #0f1e3a !important;
    color: var(--kl-lima) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 12px !important;
}

/* ── Input de texto ── */
#txt-input textarea,
#txt-input > label > textarea {
    background: #18181b !important;
    border: 1px solid #E5007D !important;
    box-shadow: 0 0 0 1px rgba(229, 0, 125, 0.3) !important;
    border-radius: 24px !important;
    color: #e2e8f0 !important;
    min-height: 44px !important;
    padding: 10px 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: none !important;
}
#txt-input textarea::placeholder {
    color: #4b5563 !important;
}
#txt-input textarea:focus,
#txt-input > label > textarea:focus {
    border-color: var(--kl-magenta) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(229, 0, 125, 0.15) !important;
}
#txt-input,
#txt-input > label,
#txt-input .block,
#txt-input > div,
div:has(> #txt-input),
div#txt-input.block.padded,
#txt-input.block.padded,
#txt-input.padded {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Botón consultar ── */
#btn-consultar button,
#btn-consultar > button,
#btn-consultar button.lg {
    background: var(--kl-magenta) !important;
    border: none !important;
    border-radius: 24px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 44px !important;
    padding: 10px 22px !important;
    transition: background 0.2s, transform 0.1s !important;
}
#btn-consultar button:hover,
#btn-consultar > button:hover {
    background: #bf0069 !important;
}
#btn-consultar button:active,
#btn-consultar > button:active {
    transform: scale(0.97) !important;
}

/* ── Preguntas frecuentes ── */
.examples-holder,
.gr-examples {
    margin-top: 4px !important;
}
.examples-holder button,
.example-btn {
    background: #16213e !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 20px !important;
    color: #ffffff !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 5px 14px !important;
    transition: background 0.2s, color 0.2s, border-color 0.2s !important;
}
.examples-holder button:hover,
.example-btn:hover {
    background: var(--kl-magenta) !important;
    color: #fff !important;
    border-color: var(--kl-magenta) !important;
}
.examples-holder > .label,
.gr-examples > label {
    color: #ffffff !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'DM Sans', sans-serif !important;
    margin-bottom: 6px !important;
}

/* ── Scrollbar del chat ── */
#chatbot::-webkit-scrollbar { width: 6px; }
#chatbot::-webkit-scrollbar-track { background: #16213e; }
#chatbot::-webkit-scrollbar-thumb { background: #2d2d4e; border-radius: 3px; }
#chatbot::-webkit-scrollbar-thumb:hover { background: var(--kl-magenta); }

/* ── Barra de progreso magenta ── */
.progress-bar { background: var(--kl-magenta) !important; }
.eta-bar { background: var(--kl-magenta) !important; }
"""

# ══════════════════════════════════════════
# FORMATEAR RESPUESTA
# ══════════════════════════════════════════
def formatear_respuesta(respuesta: dict, chunks: list) -> str:
    art    = respuesta.get("articulo") or ""
    resp   = respuesta.get("respuesta") or ""
    cita   = respuesta.get("cita_textual") or ""
    accion = respuesta.get("accion_recomendada") or ""
    advert = respuesta.get("advertencia") or ""
    fuentes = " · ".join(set(c["source"].replace(".pdf", "") for c in chunks))

    md = ""
    if art:
        md += f"**📌 {art}**\n\n"
    md += f"{resp}\n"
    if cita:
        md += f"\n> *«{cita}»*\n"
    if accion:
        md += f"\n✅ **Acción recomendada:** {accion}\n"
    if advert and advert.lower() not in ("none", "null", ""):
        md += f"\n⚠️ **Advertencia:** {advert}\n"
    md += f"\n---\n`📚 Fuentes: {fuentes}`"
    return md

# ══════════════════════════════════════════
# FUNCIÓN PRINCIPAL — generadora (fix parpadeo)
# ══════════════════════════════════════════
def consultar(pregunta: str, historial: list):
    if not pregunta.strip():
        yield historial, ""
        return

    # Paso 1: mostrar mensaje del usuario + placeholder INMEDIATAMENTE
    # El chat no desaparece porque hacemos yield antes de llamar al RAG
    historial = historial + [
        ChatMessage(role="user", content=pregunta),
        ChatMessage(role="assistant", content="⏳ _Buscando en los reglamentos..._"),
    ]
    yield historial, ""

    # Paso 2: procesar (aquí tarda el RAG + Gemini)
    try:
        chunks_ret = recuperar_chunks(pregunta, col)
        respuesta  = generar_respuesta(pregunta, chunks_ret)
        texto_bot  = formatear_respuesta(respuesta, chunks_ret)
    except Exception as e:
        texto_bot = f"⚠️ **Error:** {e}"

    # Paso 3: reemplazar el placeholder con la respuesta real
    historial[-1] = ChatMessage(role="assistant", content=texto_bot)
    yield historial, ""

# ══════════════════════════════════════════
# INTERFAZ
# ══════════════════════════════════════════
with gr.Blocks(title="KonsultaBot — Konrad Lorenz") as demo:

    gr.HTML("""
    <div style="
        background: #1a1a2e;
        border-radius: 16px 16px 0 0;
        padding: 20px 24px 16px;
        border-bottom: 3px solid #E5007D;
        display: flex; align-items: center; gap: 16px;
        margin-bottom: 0;
    ">
      <div style="
          width: 52px; height: 52px;
          background: #2d2d4e;
          border-radius: 13px;
          display: flex; align-items: center; justify-content: center;
          flex-shrink: 0;
      ">
        <svg width="42" height="42" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
          <text x="4" y="34" font-family="Arial Black,sans-serif" font-size="30"
                font-weight="900" fill="#E5007D">K</text>
          <circle cx="38" cy="14" r="8" fill="none" stroke="#97C11F" stroke-width="2.5"/>
          <circle cx="38" cy="14" r="3.5" fill="#97C11F"/>
          <text x="6" y="47" font-family="Arial,sans-serif" font-size="7"
                font-weight="700" fill="#5B9EA0" letter-spacing="1.5">BOT</text>
        </svg>
      </div>
      <div>
        <div style="font-size:22px; font-weight:700; color:#fff; font-family:'DM Sans',sans-serif; letter-spacing:-0.3px;">
          Konsulta<span style="color:#E5007D;">Bot</span>
        </div>
        <div style="font-size:11px; letter-spacing:1px; text-transform:uppercase; color:#ffffff; margin-top:3px; font-family:'DM Sans',sans-serif;">
          Asistente de Reglamentos · Fundación Universitaria Konrad Lorenz
        </div>
      </div>
      <div style="margin-left:auto; display:flex; align-items:center; gap:7px;">
        <div style="
            width:9px; height:9px;
            background:#97C11F; border-radius:50%;
            box-shadow: 0 0 0 3px rgba(151,193,31,0.3);
            animation: pulse 2s infinite;
        "></div>
        <span style="font-size:12px; color:#ffffff; font-family:'DM Sans',sans-serif;">Sistema activo</span>
      </div>
    </div>
    <style>
    @keyframes pulse { 0%,100%{box-shadow:0 0 0 3px rgba(151,193,31,0.3);} 50%{box-shadow:0 0 0 6px rgba(151,193,31,0.1);} }
    </style>
    """)

    # ── gr.Chatbot con ChatMessage como valor inicial ──
    chatbot = gr.Chatbot(
        label="",
        height=450,
        elem_id="chatbot",
        value=[ChatMessage(role="assistant", content=BIENVENIDA)],
    )

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Escribe tu pregunta sobre los reglamentos...",
            show_label=False,
            scale=4,
            autofocus=True,
            elem_id="txt-input",
        )
        btn = gr.Button("Consultar 🔍", variant="primary", scale=1, elem_id="btn-consultar")

    gr.Examples(
        examples=[
            ["¿Cuántas fallas me reprueban una materia?"],
            ["¿Qué pasa si pierdo 4 materias este semestre?"],
            ["¿Puedo solicitar un examen supletorio?"],
            ["¿Qué necesito para graduarme?"],
            ["¿Cuáles son los requisitos para la Matrícula de Honor?"],
            ["¿Qué pasa si un docente comete plagio?"],
            ["¿Cuál es el horario de la cafetería?"],
        ],
        inputs=txt,
        label="Preguntas frecuentes",
    )

    gr.HTML("""
    <div style="
        background: #1a1a2e;
        border-radius: 0 0 16px 16px;
        padding: 10px 20px;
        border-top: 1px solid #2d2d4e;
        text-align: center;
        margin-top: 4px;
    ">
      <p style="color:#ffffff; font-size:11px; margin:0; font-family:'DM Sans',sans-serif;">
        Respuestas basadas exclusivamente en los reglamentos institucionales oficiales.
        Ante dudas, consulta siempre con tu Decanatura o Coordinación Académica.
      </p>
    </div>
    """)

    btn.click(
        fn=consultar,
        inputs=[txt, chatbot],
        outputs=[chatbot, txt],
        show_progress="hidden",
    )
    txt.submit(
        fn=consultar,
        inputs=[txt, chatbot],
        outputs=[chatbot, txt],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.queue()  # obligatorio para generadoras en Gradio 6
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, css=CSS)