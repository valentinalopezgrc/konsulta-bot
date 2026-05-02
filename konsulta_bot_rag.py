"""
KonsultaBot RAG — Avance 2
Estudiante: Laura Valentina López García
Materia: Desarrollo de Aplicaciones con IA — Konrad Lorenz
Pipeline: PDF → Chunks → Embeddings → ChromaDB → Gemini
"""

import os, json, time
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import chromadb
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client  = genai.Client(api_key=API_KEY)

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
# PASO 2 — CHUNKING
# ══════════════════════════════════════════
def crear_chunks(documentos):
    chunks = []
    for doc in documentos:
        texto, source = doc["text"], doc["source"]
        inicio, idx = 0, 0
        while inicio < len(texto):
            fin = inicio + CHUNK_SIZE
            fragmento = texto[inicio:fin]
            if len(fragmento) < 50: break
            chunks.append({
                "id": f"{source.replace('.pdf','')}_chunk_{idx:04d}",
                "text": fragmento,
                "source": source,
                "chunk_index": idx
            })
            inicio += CHUNK_SIZE - CHUNK_OVERLAP
            idx += 1
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Tamaño: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    return chunks

# ══════════════════════════════════════════
# PASO 3 — EMBEDDINGS con google-genai
# ══════════════════════════════════════════
def obtener_embedding(texto: str, task: str = "RETRIEVAL_DOCUMENT"):
    resp = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(task_type=task)
    )
    return resp.embeddings[0].values

# ══════════════════════════════════════════
# PASO 4 — BASE VECTORIAL con ChromaDB
# ══════════════════════════════════════════
def construir_base_vectorial(chunks):
    Path(CHROMA_DIR).mkdir(exist_ok=True)
    db  = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    total = len(chunks)
    print(f"  🔢 Vectorizando {total} chunks con text-embedding-004...")
    ids, embeddings, textos, metadatos = [], [], [], []

    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{total}] {chunk['id']}", end="\r")
        emb = obtener_embedding(chunk["text"])
        ids.append(chunk["id"])
        embeddings.append(emb)
        textos.append(chunk["text"])
        metadatos.append({"source": chunk["source"], "chunk_index": chunk["chunk_index"]})
        time.sleep(0.7)

    for i in range(0, len(ids), 50):
        col.add(
            ids=ids[i:i+50],
            embeddings=embeddings[i:i+50],
            documents=textos[i:i+50],
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
    emb = obtener_embedding(pregunta, task="RETRIEVAL_QUERY")
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
    while True:
        print()
        pregunta = input("  🎓 Tu pregunta: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() == "salir":
            print("  👋 ¡Hasta luego!")
            break
        print("  🔍 Buscando chunks relevantes...")
        chunks = recuperar_chunks(pregunta, col)
        print(f"  📦 {len(chunks)} chunks recuperados")
        print("  🤖 Generando respuesta...")
        respuesta = generar_respuesta(pregunta, chunks)
        print("\n  " + "─"*50)
        print(json.dumps(respuesta, ensure_ascii=False, indent=4))
        print("  " + "─"*50)

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    print("\n" + "█"*60)
    print("  KONSULTABOT RAG — ASISTENTE DE REGLAMENTOS ACADÉMICOS")
    print("  Fundación Universitaria Konrad Lorenz")
    print("  Pipeline: PDF → Chunks → Embeddings → ChromaDB → Gemini")
    print("  SDK: google-genai + ChromaDB (sin LangChain)")
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