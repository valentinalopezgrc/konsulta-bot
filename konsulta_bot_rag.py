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
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 5

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
    ("¿Cuántas fallas me reprueban?",
     '{"articulo":"Art. 44","respuesta":"Máximo 20% de inasistencias. Si superas ese límite la asignatura se califica con cero.","cita_textual":"El número máximo de fallas será de un 20% sobre el total de horas.","accion_recomendada":"Lleva conteo personal de asistencias.","advertencia":null}'),
    ("Perdí 4 materias, ¿qué pasa?",
     '{"articulo":"Art. 76","respuesta":"Perder 4 o más asignaturas implica pérdida automática del cupo.","cita_textual":"Quien en un mismo período pierde cuatro (4) o más asignaturas pierde el cupo.","accion_recomendada":"Solicita reingreso al Consejo de Facultad (Art. 77).","advertencia":"Situación crítica — consultar urgentemente al Consejo de Facultad"}'),
    ("¿Cuál es el horario de la cafetería?",
     '{"articulo":null,"respuesta":"No encontré información sobre eso en los reglamentos académicos.","cita_textual":null,"accion_recomendada":"Consulta directamente a la universidad.","advertencia":null}'),
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
    print(f"\n{Fore.WHITE}  Bienvenido(a). Soy {Fore.MAGENTA}KonsultaBot{Fore.WHITE}, tu asistente virtual")
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
            print(f"  {Fore.WHITE}«{cita[:200]}»{Style.RESET_ALL}")

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