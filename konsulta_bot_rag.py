"""
KonsultaBot RAG — Avance 2
Estudiante: Laura Valentina López García
Materia: Desarrollo de Aplicaciones con IA — Konrad Lorenz
Pipeline: PDF → Chunks → Embeddings → ChromaDB → Gemini
"""

import os, json
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client  = genai.Client(api_key=API_KEY)
MODELO_ST = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

PDF_DIR       = Path("pdfs")
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "konsulta_reglamentos"
CHUNK_SIZE    = 800  
CHUNK_OVERLAP = 100  
TOP_K         = 7     

# ══════════════════════════════════════════
# PASO 1 — CARGAR PDFs
# ══════════════════════════════════════════
def cargar_pdfs(pdf_dir: Path):
    archivos = list(pdf_dir.glob("*.pdf"))
    if not archivos:
        raise FileNotFoundError(f"No hay PDFs en '{pdf_dir}/'")
    documentos = []
    for path in archivos:
        print(f"  📄 Cargando: {path.name}")
        reader = pypdf.PdfReader(str(path))
        texto = ""
        for pagina in reader.pages:
            t = pagina.extract_text()
            if t: texto += t + "\n"
        documentos.append({"source": path.name, "text": texto.strip()})
        print(f"     ✅ {len(reader.pages)} páginas | {len(texto):,} caracteres")
    return documentos

# ══════════════════════════════════════════
# PASO 2 — CHUNKING con LangChain
# ══════════════════════════════════════════
def crear_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for doc in documentos:
        if not doc["text"]:
            continue
        docs_lc = splitter.create_documents(
            [doc["text"]],
            metadatas=[{"source": doc["source"]}]
        )
        for idx, d in enumerate(docs_lc):
            chunks.append({
                "id": f"{doc['source'].replace('.pdf','')}_chunk_{idx:04d}",
                "text": d.page_content,
                "source": doc["source"],
                "chunk_index": idx
            })
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Tamaño: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    return chunks

# ══════════════════════════════════════════
# PASO 3 — EMBEDDINGS locales con SentenceTransformers
# ══════════════════════════════════════════
def obtener_embedding(texto: str, task: str = "RETRIEVAL_DOCUMENT"):
    return MODELO_ST.encode(texto).tolist()

# ══════════════════════════════════════════
# PASO 4 — BASE VECTORIAL con ChromaDB
# ══════════════════════════════════════════
def construir_base_vectorial(chunks):
    Path(CHROMA_DIR).mkdir(exist_ok=True)
    db  = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    total = len(chunks)
    print(f"  🔢 Vectorizando {total} chunks localmente...")
    textos_todos = [c["text"] for c in chunks]
    embeddings_todos = MODELO_ST.encode(textos_todos, show_progress_bar=True).tolist()
    ids = [c["id"] for c in chunks]
    metadatos = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
    for i in range(0, len(ids), 50):
        col.add(
            ids=ids[i:i+50],
            embeddings=embeddings_todos[i:i+50],
            documents=textos_todos[i:i+50],
            metadatas=metadatos[i:i+50]
        )
    print(f"\n  ✅ Base vectorial lista: {col.count()} chunks indexados")
    print(f"  📁 Guardada en: {CHROMA_DIR}")
    return col

def cargar_base_vectorial_existente():
    db  = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_collection(COLLECTION)
    print(f"  ⚡ Base vectorial existente cargada ({col.count()} chunks)")
    return col

# ══════════════════════════════════════════
# PASO 5 — RETRIEVAL
# ══════════════════════════════════════════
def recuperar_chunks(pregunta: str, col):
    emb = obtener_embedding(pregunta)
    res = col.query(query_embeddings=[emb], n_results=TOP_K,
                    include=["documents","metadatas","distances"])
    chunks = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({"text": doc, "source": meta["source"],
                       "similarity": round(1-dist, 4)})
    return chunks

# ══════════════════════════════════════════
# PASO 6 y 7 — PROMPT AUMENTADO + GEMINI
# ══════════════════════════════════════════
SYSTEM_PROMPT = """Eres KonsultaBot, un asistente experto en los Reglamentos Académicos de la Fundación Universitaria Konrad Lorenz.

Responde ÚNICAMENTE con base en el contexto recuperado entre <CONTEXTO_RAG> y </CONTEXTO_RAG>.
Nunca inventes artículos, porcentajes ni fechas que no estén en ese contexto.

Responde SIEMPRE con un JSON válido con exactamente estos 5 campos, sin texto fuera del JSON:
{
  "articulo": "Art. XX o null",
  "respuesta": "Explicación clara para el estudiante",
  "cita_textual": "Fragmento literal del reglamento o null",
  "accion_recomendada": "Qué debe hacer / a quién acudir",
  "advertencia": "Riesgo académico relevante o null"
}"""
FEW_SHOT = [
    # PREGRADO / INSTITUCIONAL — Fallas
    (
        "¿Cuántas fallas me reprueban una materia?",
        '{"articulo":"Art. 43, Párrafo 1","respuesta":"El máximo de ausencias permitido en una asignatura es del 20% sobre el total de horas del periodo académico. Si igualas o superas ese porcentaje, la asignatura se califica automáticamente con cero y queda registrada como reprobada por inasistencia.","cita_textual":"El número máximo de ausencias que un estudiante puede acumular en una asignatura será de un 20% sobre el total de horas que la asignatura tiene en el periodo académico.","accion_recomendada":"Lleva un conteo personal de tus asistencias desde el primer día. Si tienes una situación médica o de fuerza mayor, preséntate ante el Decano (Art. 43, Párrafo 5).","advertencia":"Aunque pierdas la asignatura por fallas, puedes seguir asistiendo para realimentación, pero esa nota no se registra formalmente."}'
    ),
    # INSTITUCIONAL — Pérdida de cupo
    (
        "Perdí 4 materias este semestre, ¿pierdo el cupo?",
        '{"articulo":"Art. 79","respuesta":"Sí. Quien en un mismo periodo repruebe cuatro o más asignaturas pierde el cupo en el programa que esté cursando.","cita_textual":"Quien en un mismo periodo repruebe cuatro (4) o más asignaturas, perderá el cupo en el programa que esté cursando.","accion_recomendada":"Puedes solicitar reintegro ante el Consejo de Facultad (Art. 80), pero es obligatorio haber recibido asesoría del Centro de Consejería Académica.","advertencia":"Situación crítica. Si es la segunda vez que pierdes el cupo, el reintegro lo decide el Consejo Académico por única vez."}'
    ),
    # PREGRADO — Supletorio
    (
        "No pude presentar un parcial por enfermedad, ¿puedo pedir supletorio?",
        '{"articulo":"Art. 65 y Art. 66","respuesta":"Sí, puedes solicitar un examen supletorio si la causa tiene carácter de fuerza mayor verificable. Tienes tres días hábiles desde la fecha del parcial para presentar la solicitud escrita y soportada ante la Coordinación Académica.","cita_textual":"La razón válida para autorizar la presentación de un supletorio debe tener el carácter de fuerza mayor verificable, es decir, ser totalmente ajena a la voluntad y decisión del estudiante.","accion_recomendada":"1) Consigue la excusa médica. 2) Presenta solicitud escrita a la Coordinación dentro de los 3 días hábiles. 3) Paga los derechos pecuniarios antes de la prueba.","advertencia":"La excusa NO retira las fallas de asistencia acumuladas. Si no asistes al supletorio autorizado, pierdes el derecho y te califican con cero."}'
    ),
    # POSGRADO — Fallas
    (
        "¿Cuántas fallas me reprueban en posgrado?",
        '{"articulo":"Art. 31, Parágrafo 1","respuesta":"En posgrado el máximo de inasistencias es del 15% sobre el total de horas de la asignatura. Si iguala o supera ese porcentaje, la asignatura se califica con cero y queda reprobada por inasistencia.","cita_textual":"El número máximo de fallas que un estudiante puede acumular en una asignatura será de un 15% sobre el total de horas que la asignatura tiene en el período académico.","accion_recomendada":"Lleva conteo personal de asistencias. El Decano de Posgrados tiene discrecionalidad en situaciones de fuerza mayor (Art. 31, Parágrafo 5).","advertencia":"Las excusas médicas NO retiran las fallas — solo habilitan supletorios. La asignatura perdida por inasistencia debe cursarse como repitente."}'
    ),
    # DOCENTE — Fraude
    (
        "¿Qué pasa si un profesor comete plagio?",
        '{"articulo":"Art. 42 literal i) — Reglamento Docente","respuesta":"Apropiarse de trabajos ajenos o faltar a la integridad académica es una falta disciplinaria explícita. Según su gravedad puede sancionarse con amonestación escrita, suspensión sin remuneración de 3 a 8 días, o terminación del contrato.","cita_textual":"Faltar a la integridad académica, apropiarse o aprovecharse en forma indebida de trabajos de investigación, textos, artículos, obras o materiales de propiedad de otros autores.","accion_recomendada":"La investigación la abre la Decanatura (Art. 46). El docente puede interponer recurso de reposición ante la Decanatura y apelación ante la Vicerrectoría Académica.","advertencia":"El Reglamento Docente vigente es el Acuerdo No. 19 de septiembre 2025."}'
    ),
    # FUERA DE DOMINIO
    (
        "¿Cuál es el horario de la cafetería?",
        '{"articulo":null,"respuesta":"Esa información no se encuentra en los reglamentos institucionales disponibles.","cita_textual":null,"accion_recomendada":"Consulta directamente la página web de la Konrad Lorenz o comunícate con la administración del campus.","advertencia":null}'
    ),
]

def generar_respuesta(pregunta: str, chunks) -> dict:
    contexto = "\n\n---\n\n".join([
        f"[{c['source']} | similitud={c['similarity']}]\n{c['text']}"
        for c in chunks
    ])
    prompt = SYSTEM_PROMPT + "\n\n"
    for p, r in FEW_SHOT:
        prompt += f"Usuario: {p}\nKonsultaBot: {r}\n\n"
    prompt += f"<CONTEXTO_RAG>\n{contexto}\n</CONTEXTO_RAG>\n"
    prompt += f"Usuario: {pregunta}\nKonsultaBot (solo JSON):"
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw  = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except:
        return {"articulo":None,"respuesta":raw,"cita_textual":None,
                "accion_recomendada":None,"advertencia":"Error JSON"}

# ══════════════════════════════════════════
# LOOP INTERACTIVO
# ══════════════════════════════════════════
def loop_interactivo(col):
    from colorama import Fore, Back, Style, init
    init()

    # ── Encabezado ──
    print(f"\n{Fore.LIGHTGREEN_EX}{'━'*60}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{Back.MAGENTA}{'  🎓 KONSULTABOT'.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'  Asistente de Reglamentos Institucionales'.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'  Fundación Universitaria Konrad Lorenz'.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}{'━'*60}{Style.RESET_ALL}")

    # ── Saludo ──
    print(f"\n{Fore.WHITE}  Bienvenido(a)! Soy {Fore.MAGENTA}KonsultaBot{Fore.WHITE}, tu asistente virtual")
    print(f"  especializado en los reglamentos institucionales de la Fundación Universitaria Konrad Lorenz.{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}  Estoy aquí para ayudarte con información relacionada a:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}     📘 Reglamento Académico de Pregrado{Style.RESET_ALL}")
    print(f"{Fore.WHITE}     📗 Reglamento Académico de Posgrado{Style.RESET_ALL}")
    print(f"{Fore.WHITE}     📙 Reglamento Docente{Style.RESET_ALL}")
    print(f"{Fore.WHITE}     📕 Reglamento Académico Institucional{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}  Haz tu pregunta y consultaré la información en las fuentes oficiales.{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}  💡 Escribe 'salir' para finalizar la sesión.{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'━'*60}{Style.RESET_ALL}\n")

    while True:
        pregunta = input(f"\n{Fore.CYAN}  🎓 Tu pregunta: {Style.RESET_ALL}").strip()
        if not pregunta:
            continue
        if pregunta.lower() == "salir":
            print(f"\n{Fore.BLUE}  👋 Sesión finalizada. Gracias por usar KonsultaBot.{Style.RESET_ALL}\n")
            break

        print(f"\n{Fore.YELLOW}  🔍 Buscando en los reglamentos...{Style.RESET_ALL}")
        chunks = recuperar_chunks(pregunta, col)

        print(f"\n{Fore.YELLOW}  🤖 Generando respuesta...{Style.RESET_ALL}")
        respuesta = generar_respuesta(pregunta, chunks)

        art      = str(respuesta.get('articulo') or 'No especificado')
        resp_txt = str(respuesta.get('respuesta') or '')
        cita     = str(respuesta.get('cita_textual') or '')
        accion   = str(respuesta.get('accion_recomendada') or '')
        advert   = str(respuesta.get('advertencia') or '')

        print(f"\n{Fore.BLUE}{'━'*60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Back.BLUE}{'  📋 RESPUESTA'.center(60)}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'━'*60}{Style.RESET_ALL}")

        print(f"\n  {Fore.CYAN}📌 Artículo:{Style.RESET_ALL} {Fore.WHITE}{art}{Style.RESET_ALL}")
        print(f"\n  {Fore.CYAN}💬 Respuesta:{Style.RESET_ALL}")
        for linea in resp_txt.split('. '):
            if linea:
                print(f"  {Fore.WHITE}{linea.strip()}.{Style.RESET_ALL}")

        if cita:
            print(f"\n  {Fore.CYAN}📝 Cita textual:{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}«{cita[:500]}»{Style.RESET_ALL}")

        if accion:
            print(f"\n  {Fore.GREEN}✅ Acción recomendada:{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}{accion}{Style.RESET_ALL}")

        if advert and advert != 'None':
            print(f"\n  {Fore.MAGENTA}⚠️  Advertencia:{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}{advert}{Style.RESET_ALL}")

        print(f"\n{Fore.MAGENTA}{'━'*60}{Style.RESET_ALL}")

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    print("\n" + "█"*60)
    print("  KONSULTABOT RAG — ASISTENTE DE REGLAMENTOS INSTITUCIONALES")
    print("  Fundación Universitaria Konrad Lorenz")
    print("  Pipeline: PDF → Chunks → Embeddings → ChromaDB → Gemini")
    print("  Embeddings: SentenceTransformers (local) | LLM: Gemini")
    print("█"*60)
    chroma_existe = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
    if chroma_existe:
        print("\n[1/4] ⚡ Base vectorial existente detectada...")
        col = cargar_base_vectorial_existente()
    else:
        print("\n[1/4] 📚 CARGANDO PDFs...")
        docs   = cargar_pdfs(PDF_DIR)
        print("\n[2/4] ✂️  CREANDO CHUNKS...")
        chunks = crear_chunks(docs)
        print("\n[3/4] 🗄️  VECTORIZANDO Y CONSTRUYENDO BASE VECTORIAL...")
        col    = construir_base_vectorial(chunks)
    print("\n[4/4] ✅ SISTEMA RAG LISTO\n")
    loop_interactivo(col)

if __name__ == "__main__":
    main()