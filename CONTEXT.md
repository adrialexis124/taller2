# CONTEXT.md — Contexto Completo del Proyecto: Lab 02 RAG con Vector Search y Evaluación

> **Este archivo contiene la documentación exhaustiva, la arquitectura, la explicación detallada de cada archivo y el código fuente completo de todo el repositorio del Lab 02.** Está diseñado para proporcionar un contexto total y autosuficiente a modelos de lenguaje (como Gemini, Claude, GPT, etc.), desarrolladores y docentes.

---

## 📋 1. Visión General del Proyecto

- **Curso / Maestría:** Maestría en Inteligencia Artificial / Asignatura de Agentes (Semana 2 - Lab 02).
- **Nombre:** Lab 02 — RAG con Vector Search y Evaluación.
- **Propósito:** Proveer la infraestructura base (andamiaje) para un sistema RAG (Retrieval-Augmented Generation) avanzado utilizando búsqueda vectorial densa con **Qdrant**, embeddings multilingües (**bge-m3** servido vía Ollama en la H200 de la USFQ, OpenAI o local MiniLM), fragmentación inteligente a nivel de tokens del modelo, detección de fallas comunes y evaluación de métricas de recuperación (**Hit Rate@k**, **MRR**) y abstención (**Abstención Correcta** y **Abstención Indebida**).

---

## 🗂️ 2. Estructura del Repositorio

```text
Lab-02-RAG-VectorSearch/
├── README.md                    # Documentación rápida y guía del taller
├── requirements.txt            # Dependencias fijadas y librerías necesarias
├── ingestion.py                # Módulo de ingesta, parseo de PDF/TXT/MD y fragmentación por TOKENS
├── rag_pipeline.py             # Pipeline completo: Ingesta → Embeddings → Qdrant → Retrieval → Prompt → Generación
├── evaluation.py               # Evaluador de Hit Rate, MRR, Abstención (Correcta e Indebida) y generación de resultados.csv
├── golden_set_plantilla.json   # Plantilla base estructurada para 10 preguntas evaluables
├── golden_ejemplo.json         # Golden set de prueba funcionando con el corpus de ejemplo (6 preguntas)
├── CONTEXT.md                  # (Este archivo) Documentación y código completo para LLMs
├── taller-02-rag-vector-search.pdf # Guía oficial del taller, enunciados, rúbrica y tabla semestral de modelos
└── ejemplos/                   # Corpus mínimo de pruebas (incluye falla por PDF escaneado)
    ├── instructivo_escaneado.pdf # PDF de prueba escaneado (sin capa de texto)
    ├── nimbus_vacaciones.md     # Documento Markdown: Política de vacaciones
    ├── nimbus_gastos.md         # Documento Markdown: Política de gastos
    └── nimbus_remoto.md         # Documento Markdown: Política de trabajo remoto
```

---

## 🔬 3. Principios de Arquitectura y Conceptos Clave

1. **Fragmentación por TOKENS del Modelo (no palabras):**
   - Los modelos de embeddings tienen un límite duro de secuencia (`max_seq_tokens`).
   - `ingestion.py` utiliza el tokenizador exacto del modelo en uso (ej. `BAAI/bge-m3` o `tiktoken` para OpenAI) para calcular y dividir en fragmentos de tamaño `chunk_tokens` con solapamiento `overlap_tokens`.
   - Evita el truncado silencioso que ocurre al fragmentar por palabras o cuando se supera la ventana del modelo.

2. **Detección Automática de PDFs Escaneados (Parte 0.a):**
   - PDFs sin capa de texto devuelven texto vacío `""`.
   - `ingestion.py` evalúa los caracteres alfanuméricos útiles (`caracteres_utiles()`). Si un documento tiene menos de 200 caracteres útiles, avisa por `sys.stderr` y **NO se indexa**, evitando generar chunks vacíos con solo etiquetas de página (`[page=1]`).

3. **Manejo Estricto del Truncado de Secuencia (Parte 0.b):**
   - El backend por defecto (`h200` / `bge-m3`) admite 8192 tokens. Se envía la bandera `truncate: false` a la API de Ollama para que, si un texto excede la ventana, lance un error explícito en lugar de recortarlo en silencio.
   - Para la Parte 0.b del taller, se provee el backend `local` (`MiniLM`), el cual trunca a 128 tokens para demostrar este modo de falla.

4. **Preguntas Negativas y Medición de Abstención (Parte 0.c y Parte 2):**
   - Las preguntas cuya respuesta NO está en el corpus (negativas) **NO entran en el denominador del Hit Rate ni del MRR** (de lo contrario un sistema perfecto obtendría máximo 0.70 de Hit Rate si 3 de 10 preguntas son negativas).
   - Se define una constante estricta de abstención: `ABSTENCION = "El corpus no contiene información suficiente."`.
   - La función `se_abstuvo()` compara el texto normalizado (minúsculas, sin tildes, sin puntuación) para evitar falsos negativos por variantes de formato.
   - Se miden dos tasas:
     - **Abstención Correcta:** Fracción de preguntas negativas donde el sistema efectivamente se abstuvo.
     - **Abstención Indebida:** Fracción de preguntas respondibles donde el sistema se abstuvo por error.

5. **Aislamiento y Selección Modular de LLM / Embeddings:**
   - **Embeddings (`EMBEDDING_BACKEND`):** `h200` (BAAI/bge-m3 en Ollama USFQ), `openai` (text-embedding-3-small) o `local` (MiniLM para Parte 0).
   - **Base de Datos Vectorial (`QDRANT_URL`):** Soporta tanto servidor Docker (`http://localhost:6333`) como ejecución local en memoria (`:memory:`).
   - **Generación:** Selecciona automáticamente entre `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, Ollama local o `Modo Inspección` (este último devuelve solo el prompt sin gastar tokens/API keys).

---

## 🛠️ 4. Guía de Instalación y Ejecución

### 1. Setup del Entorno
```bash
pip install -r requirements.txt
cp .env.example .env   # Ajustar OPENAI_API_KEY u otras claves si aplica
```

### 2. Ejecutar la Parte 0 (Demostración de Fallas en Memoria)
```bash
EMBEDDING_BACKEND=local QDRANT_URL=":memory:" python rag_pipeline.py "¿cuántos días de vacaciones puedo transferir?"
```

### 3. Ejecutar Pipeline en Producción / Baseline
```bash
python rag_pipeline.py "¿Cuál es el límite de gastos de alimentación?"
```

### 4. Evaluación Completa (Generando métricas y `resultados.csv`)
```bash
# Evaluación solo de recuperación (sin gastar créditos LLM)
python evaluation.py --corpus ejemplos --golden golden_ejemplo.json --k 5 --sin-generar

# Evaluación completa (recuperación + generación + abstención)
python evaluation.py --corpus ejemplos --golden golden_ejemplo.json --k 5
```

---

## 📄 5. Contenido y Código Fuente de Todos los Archivos

### 5.1. `requirements.txt`
```ini
# Versiones mínimas fijadas el 2026-09-19: la API de Qdrant cambió entre versiones
# (search → query_points) y el lab y los notebooks tienen que hablar la misma.
qdrant-client>=1.12
sentence-transformers>=3.0   # tokenizador de bge-m3 y el MiniLM de la Parte 0
tiktoken>=0.7                 # tokenizador de los embeddings de OpenAI
numpy>=1.26
pypdf>=4.0
python-dotenv>=1.0
openai>=1.0
anthropic>=0.30
rank-bm25>=0.2
ragas>=0.2
```

---

### 5.2. `ingestion.py`
```python
"""
Lab 02 — Ingesta y fragmentación baseline para RAG.

Este archivo evita que el lab empiece desde boilerplate. Los estudiantes deben ajustar
parseo, fragmentación y metadatos al corpus elegido.

Dos comprobaciones que existen porque sus fallas **no fallan**:

1. **Un PDF sin capa de texto** (escaneado, fotografiado) devuelve `""` en cada página; sin
   comprobación, el documento entero se convertiría en UN fragmento hecho solo de
   marcadores `[page=n]`, sin ninguna excepción, y el documento sencillamente no estaría.
   `read_document` cuenta los caracteres útiles y `load_corpus` **avisa y no indexa** lo
   que no tiene texto (Parte 0.a del taller).
2. **Los fragmentos se miden en tokens del modelo**, no en palabras, porque el modelo de
   embeddings tiene un tope en **tokens**. `rag_pipeline.py` le pasa a `load_corpus` el
   tokenizador y el tope del codificador en uso —8192 para `bge-m3`, 128 para el MiniLM de
   la Parte 0.b— y la ingesta avisa si se pide un fragmento que no cabe.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
import sys

from pypdf import PdfReader

MARCADOR_PAGINA = re.compile(r"\[page=\d+\]")
MIN_CARACTERES_UTILES = 200   # por documento; por debajo, casi seguro es un escaneo
MIN_CARACTERES_FRAGMENTO = 20  # un fragmento con menos que esto son marcadores o ruido
CHUNK_TOKENS_POR_DEFECTO = 512  # notas del curso, §4.2; acotado por el tope del modelo


@dataclass
class Chunk:
    id: str
    text: str
    document: str
    chunk_index: int
    metadata: dict


@dataclass
class Documento:
    name: str
    path: str
    text: str
    pages: int
    useful_chars: int

    @property
    def parece_escaneado(self) -> bool:
        return self.useful_chars < MIN_CARACTERES_UTILES


def caracteres_utiles(text: str) -> int:
    """Letras y dígitos, sin contar los marcadores de página que nosotros mismos añadimos."""
    return sum(ch.isalnum() for ch in MARCADOR_PAGINA.sub("", text))


def read_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[page={page_number}]\n{text}")
    return "\n".join(pages), len(reader.pages)


def read_document(path: Path) -> Documento:
    if path.suffix.lower() == ".pdf":
        text, pages = read_pdf(path)
    elif path.suffix.lower() in {".txt", ".md"}:
        text, pages = path.read_text(encoding="utf-8"), 1
    else:
        raise ValueError(f"Formato no soportado: {path}")
    return Documento(name=path.name, path=str(path), text=text, pages=pages,
                     useful_chars=caracteres_utiles(text))


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fixed_size_chunks_tokens(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Fragmentos de `chunk_tokens` tokens del modelo, con solapamiento, decodificados a texto.

    La unidad es el token del **mismo** tokenizador que va a vectorizar: así `chunk_tokens`
    y `max_seq_length` se comparan en la misma escala y el truncado deja de ser invisible.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens debe ser menor que chunk_tokens")
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunks.append(tokenizer.decode(ids[start:end]).strip())
        if end == len(ids):
            break
        start = end - overlap_tokens
    return chunks


def fixed_size_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """La versión en PALABRAS. Se conserva para que se pueda medir la diferencia con la
    versión en tokens; no es el baseline."""
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def load_corpus(corpus_dir: Path, tokenizer=None, max_seq_tokens: int | None = None,
                chunk_tokens: int | None = None, overlap_tokens: int | None = None,
                verbose: bool = True) -> list[Chunk]:
    """Lee los documentos del corpus y los fragmenta en tokens del modelo.

    - `tokenizer` y `max_seq_tokens` los pasa `rag_pipeline.py` desde el modelo cargado.
      Sin tokenizador cae a la versión en palabras y lo dice, porque entonces nadie sabe
      cuánto se trunca.
    - `chunk_tokens` por defecto es `min(512, max_seq_tokens - 2)`: 512 tokens, el tamaño de
      las notas del curso, salvo que el modelo no los admita —el MiniLM de la Parte 0.b ve
      126, su tope menos los dos tokens especiales—. El solapamiento por defecto es un quinto
      del fragmento. Pedir más que el tope produce un AVISO, no una excepción, porque puede
      ser una decisión deliberada que hay que justificar en el informe.
    - Un documento con menos de MIN_CARACTERES_UTILES caracteres útiles **no se indexa** y
      se avisa: es el PDF escaneado de la Parte 0.a del Taller 2.
    """
    if chunk_tokens is None:
        chunk_tokens = max(32, min(CHUNK_TOKENS_POR_DEFECTO, (max_seq_tokens or 128) - 2))
    if overlap_tokens is None:
        overlap_tokens = chunk_tokens // 5
    if max_seq_tokens and chunk_tokens > max_seq_tokens - 2 and verbose:
        print(f"AVISO ingesta: chunk_tokens={chunk_tokens} supera el tope del modelo "
              f"({max_seq_tokens} tokens): lo que pase de {max_seq_tokens - 2} tokens no "
              f"llega al índice —el MiniLM lo trunca en silencio; la H200 rechaza la "
              f"petición—.", file=sys.stderr)

    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        doc = read_document(path)
        if doc.parece_escaneado:
            if verbose:
                print(f"AVISO ingesta: {doc.name}: {doc.useful_chars} caracteres útiles en "
                      f"{doc.pages} página(s). ¿Escaneado o fotografiado? NO se indexa: sin "
                      f"texto no hay nada que recuperar (OCR aparte).", file=sys.stderr)
            continue
        text = normalize_text(doc.text)
        if tokenizer is not None:
            piezas = fixed_size_chunks_tokens(text, tokenizer, chunk_tokens, overlap_tokens)
            estrategia = {"strategy": "fixed_size", "unit": "tokens",
                          "chunk_tokens": chunk_tokens, "overlap_tokens": overlap_tokens,
                          "max_seq_tokens": max_seq_tokens}
        else:
            if verbose:
                print("AVISO ingesta: sin tokenizador, fragmentando en PALABRAS: la unidad no es "
                      "la del modelo y el truncado no se puede medir.", file=sys.stderr)
            piezas = fixed_size_chunks(text)
            estrategia = {"strategy": "fixed_size", "unit": "words",
                          "chunk_size": 900, "overlap": 150}
        idx = 0
        for chunk_text in piezas:
            if caracteres_utiles(chunk_text) < MIN_CARACTERES_FRAGMENTO:
                continue   # solo marcadores de página o espacio: no se indexa
            chunks.append(Chunk(id=f"{path.stem}-{idx:04d}", text=chunk_text,
                                document=path.name, chunk_index=idx,
                                metadata={"source_path": str(path), **estrategia}))
            idx += 1
        if verbose:
            print(f"ingesta: {doc.name}: {doc.pages} página(s), {doc.useful_chars} caracteres "
                  f"útiles → {idx} fragmento(s)")
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    tokenizer, tope = None, None
    try:
        from rag_pipeline import crear_codificador
        codificador = crear_codificador()
        tokenizer, tope = codificador.tokenizer, codificador.max_seq_tokens
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        print(f"(sin modelo de embeddings: {exc}; se fragmenta en palabras)", file=sys.stderr)
    chunks = load_corpus(corpus, tokenizer=tokenizer, max_seq_tokens=tope)
    write_chunks_jsonl(chunks, Path("data/chunks.jsonl"))
    print(f"Chunks generados: {len(chunks)}")
```

---

### 5.3. `rag_pipeline.py`
```python
"""
Lab 02 — Pipeline RAG baseline.

Ingesta → embeddings → Qdrant → recuperación → prompt → generación. La generación queda
aislada para que la recuperación pueda evaluarse sin gastar tokens.

Cinco decisiones que conviene conocer antes de tocar nada:

1. **Los embeddings son de `bge-m3`, servido en la H200 de la USFQ** (fila
   `embed_local_multilingue` de la tabla semestral: multilingüe, 1024 dimensiones, tope de
   8192 tokens). Hace falta estar en la red de la universidad —GlobalProtect conectada—. Sin
   VPN, la alternativa es la API de OpenAI (`EMBEDDING_BACKEND=openai`, fila
   `embed_api_economico`; `EMBEDDING_MODEL` elige otro de sus modelos). La ruta `local`
   —el MiniLM de los notebooks, tope de 128— existe **solo para la Parte 0.b del taller**.
2. **El tope de tokens se lee, no se supone**, y la ingesta fragmenta en tokens del **mismo**
   tokenizador que vectoriza, así que `chunk_tokens` y el tope se comparan en la misma escala.
   Contra la H200 se pide además `truncate: false`: si un texto no cabe, el servidor da error
   en vez de recortarlo en silencio.
3. **Una sola frase de abstención**, `ABSTENCION`, compartida con el golden set y con el
   notebook del miércoles, y detectada **normalizada** (`se_abstuvo`).
4. **La generación elige su ruta por la clave disponible**: con `OPENAI_API_KEY`, la fila
   `propietario_economico`; con `ANTHROPIC_API_KEY`, `juez_economico`; sin ninguna, Ollama
   con `open_weight_pequeno`; y si tampoco hay Ollama, «modo inspección», que devuelve el
   prompt para que la recuperación se pueda depurar igual.
5. **La API de Qdrant es la del notebook del martes** —`create_collection` +
   `query_points`—, y `QDRANT_URL=":memory:"` corre sin Docker.

Ningún nombre de modelo se escribe aquí a mano: salen de `fuentes/modelos/modelos-2026-1.json`
por su `id`, y las variables de entorno solo los sobreescriben.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import unicodedata

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import numpy as np

from ingestion import Chunk, load_corpus

load_dotenv()

REPO = Path(__file__).resolve().parents[3]
TABLA_MODELOS = REPO / "fuentes" / "modelos" / "modelos-2026-1.json"


def fila_de_la_tabla(clave: str, id_: str) -> dict:
    """Una fila de la tabla semestral por su `id`. Si el lab se copió fuera del repositorio
    y la tabla no está, devuelve {} y las variables de entorno mandan."""
    try:
        datos = json.loads(TABLA_MODELOS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return next((f for f in datos.get(clave, []) if f.get("id") == id_), {})


# ── Abstención: una frase, comparada normalizada ─────────────────────────────────────
ABSTENCION = "El corpus no contiene información suficiente."


def _normalizar_texto(t: str) -> str:
    """minúsculas, sin tildes, sin puntuación: 'informacion suficiente' e
    'información suficiente.' son la misma frase."""
    t = unicodedata.normalize("NFKD", t.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).split())


def se_abstuvo(respuesta: str) -> bool:
    return _normalizar_texto(ABSTENCION) in _normalizar_texto(respuesta or "")


# ── Configuración ────────────────────────────────────────────────────────────────────
COLLECTION = os.getenv("QDRANT_COLLECTION", "mmia6013_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")   # ":memory:" corre sin Docker
GENERATION_MODEL = os.getenv("GENERATION_MODEL")   # si no, se decide por la clave disponible
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")   # generación local

# Embeddings: h200 (por defecto) · openai (sin VPN) · local (solo la Parte 0.b del taller).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "h200").strip().lower()
H200_EMBED_URL = os.getenv("H200_EMBED_URL", "http://172.28.230.10:11434")   # Ollama de la H200
FILA_EMBEDDINGS = {"h200": "embed_local_multilingue",     # bge-m3
                   "openai": "embed_api_economico",
                   "local": "embed_notebook_s2"}          # MiniLM, tope 128: Parte 0.b
if EMBEDDING_BACKEND not in FILA_EMBEDDINGS:
    raise SystemExit(f"EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}: usa h200, openai o local")
FILA_EMBEDDING = fila_de_la_tabla("embeddings", FILA_EMBEDDINGS[EMBEDDING_BACKEND])
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or FILA_EMBEDDING.get("model")
                   or {"h200": "BAAI/bge-m3", "openai": "text-embedding-3-small",
                       "local": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
                   [EMBEDDING_BACKEND])


def _normalizar(vectores) -> np.ndarray:
    v = np.asarray(vectores, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


class _TokenizadorTiktoken:
    """Adaptador de tiktoken con la misma interfaz que usa la ingesta de un tokenizador de
    Hugging Face: `tok(texto, add_special_tokens=False, truncation=False)["input_ids"]` y
    `tok.decode(ids)`."""

    def __init__(self, modelo: str):
        import tiktoken
        try:
            self._enc = tiktoken.encoding_for_model(modelo)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def __call__(self, texto, add_special_tokens=False, truncation=False):
        return {"input_ids": self._enc.encode(texto)}

    def decode(self, ids):
        return self._enc.decode(list(ids))


class CodificadorH200:
    """`bge-m3` en el Ollama de la H200 (puerto 11434). Solo biblioteca estándar para la red.

    El tokenizador es el de `BAAI/bge-m3` en Hugging Face (se descarga solo el tokenizador,
    no el modelo), para que la ingesta cuente en los mismos tokens que el servidor.
    """

    LOTE = 32

    def __init__(self, modelo: str, tope: int):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.max_seq_tokens = int(tope)
        corto = modelo.split("/")[-1].lower()
        servidos = [m["name"] for m in self._pedir("/api/tags", timeout=10).get("models", [])]
        candidatos = sorted(n for n in servidos if n.lower().startswith(corto))
        if not candidatos:
            raise SystemExit(f"la H200 no sirve {corto} (sirve: {', '.join(servidos) or 'nada'})")
        self.id_servido = candidatos[0]

    def _pedir(self, ruta: str, cuerpo: dict | None = None, timeout: int = 180) -> dict:
        import urllib.error
        import urllib.request
        datos = None if cuerpo is None else json.dumps(cuerpo).encode()
        pet = urllib.request.Request(f"{H200_EMBED_URL}{ruta}", data=datos,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(pet, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            raise SystemExit(f"la H200 respondió {err.code} en {ruta}: "
                             f"{err.read().decode(errors='replace')[:300]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise SystemExit(
                f"la H200 no responde en {H200_EMBED_URL} ({err}). ¿Está conectada la VPN "
                "GlobalProtect? Sin VPN, usa la API de OpenAI: EMBEDDING_BACKEND=openai "
                "con OPENAI_API_KEY en tu .env.") from None

    def encode(self, textos: list[str]) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), self.LOTE):
            r = self._pedir("/api/embed", {
                "model": self.id_servido, "input": textos[i:i + self.LOTE],
                "truncate": False, "options": {"num_ctx": self.max_seq_tokens}})
            salida.extend(r["embeddings"])
        return _normalizar(salida)


class CodificadorOpenAI:
    """Embeddings por la API de OpenAI: la alternativa sin VPN. Gasta clave, poco."""

    LOTE = 64

    def __init__(self, modelo: str, tope: int):
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("EMBEDDING_BACKEND=openai necesita OPENAI_API_KEY en tu .env")
        from openai import OpenAI
        self._cliente = OpenAI()
        self.modelo = modelo
        self.tokenizer = _TokenizadorTiktoken(modelo)
        self.max_seq_tokens = int(tope)

    def encode(self, textos: list[str]) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), self.LOTE):
            r = self._cliente.embeddings.create(model=self.modelo, input=textos[i:i + self.LOTE])
            salida.extend(d.embedding for d in r.data)
        return _normalizar(salida)


class CodificadorLocal:
    """El MiniLM de los notebooks, en tu máquina. Solo para la Parte 0.b del taller: trunca a
    128 tokens sin avisar, que es justamente lo que esa parte hace ver."""

    def __init__(self, modelo: str, tope: int | None = None):
        from sentence_transformers import SentenceTransformer
        self._modelo = SentenceTransformer(modelo)
        self.tokenizer = self._modelo.tokenizer
        self.max_seq_tokens = int(self._modelo.max_seq_length)   # leído del modelo

    def encode(self, textos: list[str]) -> np.ndarray:
        return self._modelo.encode(textos, normalize_embeddings=True)


def crear_codificador():
    tope = int(FILA_EMBEDDING.get("max_seq_tokens") or 8192)
    clase = {"h200": CodificadorH200, "openai": CodificadorOpenAI,
             "local": CodificadorLocal}[EMBEDDING_BACKEND]
    return clase(EMBEDDING_MODEL, tope)


class RagPipeline:
    def __init__(self, collection: str = COLLECTION):
        self.collection = collection
        self.encoder = crear_codificador()
        self.max_seq_tokens = self.encoder.max_seq_tokens
        self.client = QdrantClient(":memory:") if QDRANT_URL == ":memory:" else QdrantClient(url=QDRANT_URL)
        self.ultima_generacion = "sin generar"

    # ── ingesta ──
    def ingest(self, corpus_dir: Path, chunk_tokens: int | None = None,
               overlap_tokens: int | None = None) -> list[Chunk]:
        """Fragmenta en tokens del modelo cargado, con su tope leído y no supuesto."""
        return load_corpus(Path(corpus_dir), tokenizer=self.encoder.tokenizer,
                           max_seq_tokens=self.max_seq_tokens,
                           chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

    # ── índice ──
    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.encoder.encode([chunk.text for chunk in chunks])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "document": chunk.document,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    # ── recuperación ──
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_vector = self.encoder.encode([question])[0].tolist()
        respuesta = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": hit.score,
                "chunk_id": hit.payload["chunk_id"],
                "document": hit.payload["document"],
                "text": hit.payload["text"],
            }
            for hit in respuesta.points
        ]

    # ── prompt ──
    @staticmethod
    def build_prompt(question: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join(
            f"[{idx}] {ctx['document']} / {ctx['chunk_id']}\n{ctx['text']}"
            for idx, ctx in enumerate(contexts, start=1)
        )
        return f"""Responde usando solo el contexto recuperado.
Si el contexto no contiene la respuesta, di exactamente: "{ABSTENCION}"

Contexto:
{context_text}

Pregunta: {question}
Respuesta:"""

    # ── generación ──
    def generate(self, prompt: str) -> str:
        """Elige la ruta por la clave disponible; sin ninguna, modo inspección."""
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            modelo = GENERATION_MODEL or fila_de_la_tabla("models", "propietario_economico").get("model")
            self.ultima_generacion = f"openai:{modelo}"
            r = OpenAI().chat.completions.create(
                model=modelo, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            modelo = GENERATION_MODEL or fila_de_la_tabla("models", "juez_economico").get("model")
            self.ultima_generacion = f"anthropic:{modelo}"
            r = anthropic.Anthropic().messages.create(
                model=modelo, max_tokens=400, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if b.type == "text")
        try:
            import urllib.request
            modelo = GENERATION_MODEL or fila_de_la_tabla("models", "open_weight_pequeno").get("model")
            cuerpo = json.dumps({"model": modelo, "prompt": prompt, "stream": False,
                                 "options": {"temperature": 0}}).encode()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=cuerpo,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.ultima_generacion = f"ollama:{modelo}"
                return json.loads(r.read())["response"]
        except Exception:  # noqa: BLE001 — sin Ollama tampoco: modo inspección
            self.ultima_generacion = "modo inspección (sin clave ni Ollama)"
            return "[modo inspección: no hay clave de API ni Ollama; este es el prompt]\n" + prompt

    def answer(self, question: str, top_k: int = 5) -> dict:
        contexts = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, contexts)
        texto = self.generate(prompt)
        inspeccion = self.ultima_generacion.startswith("modo inspección")
        return {"question": question, "contexts": contexts, "prompt": prompt,
                "answer": texto, "abstained": None if inspeccion else se_abstuvo(texto),
                "generator": self.ultima_generacion}


if __name__ == "__main__":
    import sys
    corpus = Path(os.getenv("CORPUS_DIR", "corpus"))
    pipeline = RagPipeline()
    print(f"embeddings: {EMBEDDING_BACKEND} · {EMBEDDING_MODEL} · tope {pipeline.max_seq_tokens} "
          f"tokens · Qdrant {QDRANT_URL}")
    chunks = pipeline.ingest(corpus)
    if not chunks:
        sys.exit(f"ningún fragmento indexable en {corpus}/ (¿PDFs escaneados? ¿carpeta vacía?)")
    pipeline.index(chunks)
    print(f"indexados {len(chunks)} fragmentos de {len({c.document for c in chunks})} documento(s)")
    question = " ".join(sys.argv[1:]) or "REEMPLAZAR por una pregunta de prueba"
    for hit in pipeline.retrieve(question):
        print(f"  {hit['score']:.3f}  {hit['document']} / {hit['chunk_id']}  {hit['text'][:80]}…")
    salida = pipeline.answer(question)
    print(f"\n[{salida['generator']}] abstuvo={salida['abstained']}\n{salida['answer'][:1200]}")
```

---

### 5.4. `evaluation.py`
```python
"""
Lab 02 — Métricas de recuperación y de abstención sobre el golden set.

    python evaluation.py --golden golden_set.json --k 5
    python evaluation.py --golden golden_set.json --k 3 --sin-generar     # solo recuperación
    QDRANT_URL=":memory:" python evaluation.py --corpus ejemplos --golden golden_ejemplo.json

Qué mide, y por qué así:

1. **Las negativas no entran en el Hit Rate ni en el MRR.** Una pregunta sin documento
   esperado —las negativas obligatorias, cuya respuesta correcta es abstenerse— no tiene
   nada que recuperar: si contara como fallo, con tres negativas de diez un sistema
   **perfecto** reportaría 0,70 y no podría subir.
2. **El golden set se anota por documento y por fragmento literal**, no por `chunk_id`: un
   `chunk_id` cambia al re-fragmentar, y la Opción A del taller invalidaría las anotaciones
   sin fallar. Un acierto es «algún fragmento recuperado pertenece a un documento
   fuente y, si se dio un fragmento esperado, lo contiene».
3. **La tasa de abstención se mide de verdad**, generando, y en sus dos direcciones: la
   correcta (negativas en las que el sistema se abstuvo) y la indebida (respondibles en las
   que también se abstuvo). Son dos números, no uno (sesión 09). La detección compara
   normalizado, con la misma constante que el prompt.
4. **Una fila sin rellenar se cuenta y se avisa.** Las filas de la plantilla que siguen con
   `REEMPLAZAR` no se evalúan —no hay nada que evaluar—, pero saltarlas en silencio haría
   que un golden set a medio anotar reportara métricas sobre menos preguntas de las que el
   taller pide, sin que nada lo dijera. El resumen trae `n_sin_rellenar` e
   `ids_sin_rellenar`, y la función lo imprime.
5. **Escribe `resultados.csv`** con una fila por consulta: es el entregable crudo del que se
   derivan todas las tablas del informe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from rag_pipeline import RagPipeline, se_abstuvo, _normalizar_texto


def load_golden_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def es_respondible(item: dict) -> bool:
    """Tiene al menos un documento fuente. Las negativas no; la adversarial, según el
    estudiante decida (ver la plantilla)."""
    return bool(item.get("documentos_fuente"))


def acierta(hit: dict, item: dict) -> bool:
    if hit["document"] not in item.get("documentos_fuente", []):
        return False
    esperado = (item.get("fragmento_esperado") or "").strip()
    return not esperado or _normalizar_texto(esperado) in _normalizar_texto(hit["text"])


def posicion_del_primer_acierto(hits: list[dict], item: dict) -> int | None:
    for pos, hit in enumerate(hits, start=1):
        if acierta(hit, item):
            return pos
    return None


def _media(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else math.nan


def evaluate_retrieval(golden_set: list[dict], pipeline: RagPipeline, k: int = 5,
                       generar: bool = True) -> dict:
    rows = []
    sin_rellenar = []
    for item in golden_set:
        pregunta = item["pregunta"]
        if not pregunta or pregunta.startswith("REEMPLAZAR"):
            sin_rellenar.append(item.get("id"))
            continue
        respondible = es_respondible(item)
        hits = pipeline.retrieve(pregunta, top_k=k)
        pos = posicion_del_primer_acierto(hits, item) if respondible else None
        fila = {
            "id": item["id"],
            "tipo": item.get("tipo", "sin_tipo"),
            "respondible": respondible,
            "k": k,
            "hit": (pos is not None) if respondible else None,
            "reciprocal_rank": (1 / pos if pos else 0.0) if respondible else None,
            "posicion": pos,
            "score_top1": hits[0]["score"] if hits else None,
            "retrieved_ids": [h["chunk_id"] for h in hits],
            "abstuvo": None,
            "generador": None,
        }
        if generar:
            salida = pipeline.answer(pregunta, top_k=k)
            fila["abstuvo"] = salida["abstained"]
            fila["generador"] = salida["generator"]
            fila["respuesta"] = salida["answer"][:500]
        rows.append(fila)

    respondibles = [r for r in rows if r["respondible"]]
    negativas = [r for r in rows if not r["respondible"]]
    resumen = {
        "k": k,
        "n_preguntas": len(rows),
        "n_sin_rellenar": len(sin_rellenar),
        "ids_sin_rellenar": sin_rellenar,
        "n_respondibles": len(respondibles),
        "n_negativas": len(negativas),
        "hit_rate": _media([float(r["hit"]) for r in respondibles]),
        "mrr": _media([r["reciprocal_rank"] for r in respondibles]),
        "abstencion_correcta": _media([float(r["abstuvo"]) for r in negativas if r["abstuvo"] is not None]),
        "abstencion_indebida": _media([float(r["abstuvo"]) for r in respondibles if r["abstuvo"] is not None]),
        "por_tipo": {},
        "rows": rows,
    }
    for tipo in sorted({r["tipo"] for r in rows}):
        del_tipo = [r for r in rows if r["tipo"] == tipo]
        resp = [r for r in del_tipo if r["respondible"]]
        resumen["por_tipo"][tipo] = {
            "n": len(del_tipo),
            "hit_rate": _media([float(r["hit"]) for r in resp]),
            "mrr": _media([r["reciprocal_rank"] for r in resp]),
            "abstuvo": _media([float(r["abstuvo"]) for r in del_tipo if r["abstuvo"] is not None]),
        }
    if sin_rellenar:
        print(f"AVISO: {len(sin_rellenar)} de {len(golden_set)} preguntas del golden set "
              f"siguen sin rellenar y NO se evaluaron (ids {sin_rellenar}). "
              f"Las métricas de abajo se calcularon sobre {len(rows)}.")
    return resumen


def escribir_csv(rows: list[dict], path: Path, modelo_embeddings: str) -> None:
    """Una fila por consulta: el crudo del que se derivan las tablas del informe."""
    columnas = ["id", "tipo", "respondible", "modelo_embeddings", "k", "posicion", "hit",
                "reciprocal_rank", "score_top1", "abstuvo", "generador", "retrieved_ids"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for r in rows:
            w.writerow({c: (r.get(c) if c != "modelo_embeddings" else modelo_embeddings)
                        for c in columnas} | {"retrieved_ids": " ".join(r["retrieved_ids"])})


def _fmt(x) -> str:
    return "sin medir" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("golden_set.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--csv", type=Path, default=Path("resultados.csv"))
    ap.add_argument("--sin-generar", action="store_true",
                    help="solo recuperación: no llama a ningún LLM y no mide abstención")
    ap.add_argument("--sin-indexar", action="store_true",
                    help="usa la colección ya indexada en Qdrant (no sirve con :memory:)")
    args = ap.parse_args()

    from rag_pipeline import EMBEDDING_MODEL
    pipeline = RagPipeline()
    if not args.sin_indexar:
        chunks = pipeline.ingest(args.corpus)
        if not chunks:
            raise SystemExit(f"ningún fragmento indexable en {args.corpus}/")
        pipeline.index(chunks)
    golden_set = load_golden_set(args.golden)
    report = evaluate_retrieval(golden_set, pipeline, k=args.k, generar=not args.sin_generar)
    escribir_csv(report["rows"], args.csv, EMBEDDING_MODEL)

    print(f"golden set: {report['n_preguntas']} preguntas ({report['n_respondibles']} respondibles, "
          f"{report['n_negativas']} negativas) · k = {report['k']} · modelo {EMBEDDING_MODEL}")
    print(f"Hit Rate@{args.k} (respondibles): {_fmt(report['hit_rate'])}")
    print(f"MRR (respondibles):          {_fmt(report['mrr'])}")
    print(f"abstención correcta:         {_fmt(report['abstencion_correcta'])}   (negativas)")
    print(f"abstención indebida:         {_fmt(report['abstencion_indebida'])}   (respondibles)")
    for tipo, m in report["por_tipo"].items():
        print(f"  {tipo:<12} n={m['n']}  hit={_fmt(m['hit_rate'])}  mrr={_fmt(m['mrr'])}  abstuvo={_fmt(m['abstuvo'])}")
    print(f"filas crudas en {args.csv}")
```

---

### 5.5. `golden_set_plantilla.json`
```json
[
  {
    "id": 1,
    "tipo": "simple",
    "pregunta": "REEMPLAZAR: pregunta con respuesta en un solo fragmento",
    "respuesta_esperada": "REEMPLAZAR: respuesta correcta según el documento fuente",
    "documentos_fuente": ["REEMPLAZAR.pdf"],
    "fragmento_esperado": "REEMPLAZAR: una frase LITERAL del documento donde está la respuesta",
    "criterio_exito": "El sistema recupera un fragmento del documento fuente que contiene la frase y responde sin agregar información externa."
  },
  {
    "id": 2, "tipo": "simple", "pregunta": "", "respuesta_esperada": "", "documentos_fuente": [], "fragmento_esperado": "", "criterio_exito": ""
  },
  {
    "id": 3, "tipo": "simple", "pregunta": "", "respuesta_esperada": "", "documentos_fuente": [], "fragmento_esperado": "", "criterio_exito": ""
  },
  {
    "id": 4,
    "tipo": "multi-chunk",
    "pregunta": "REEMPLAZAR: pregunta que requiere combinar varias partes del corpus",
    "respuesta_esperada": "REEMPLAZAR: respuesta que sintetiza varias fuentes",
    "documentos_fuente": ["doc1.pdf", "doc2.pdf"],
    "fragmento_esperado": "",
    "criterio_exito": "El sistema recupera al menos un fragmento de alguno de los documentos fuente en el top-k y cita ambas fuentes en la respuesta."
  },
  {
    "id": 5, "tipo": "multi-chunk", "pregunta": "", "respuesta_esperada": "", "documentos_fuente": [], "fragmento_esperado": "", "criterio_exito": ""
  },
  {
    "id": 6, "tipo": "multi-chunk", "pregunta": "", "respuesta_esperada": "", "documentos_fuente": [], "fragmento_esperado": "", "criterio_exito": ""
  },
  {
    "id": 7,
    "tipo": "negativo",
    "pregunta": "REEMPLAZAR: pregunta cuya respuesta NO está en el corpus",
    "respuesta_esperada": "El corpus no contiene información suficiente.",
    "documentos_fuente": [],
    "fragmento_esperado": "",
    "criterio_exito": "No hay nada que recuperar: no entra en el Hit Rate ni en el MRR. Lo que se mide es si el sistema se abstiene (tasa de abstención correcta)."
  },
  {
    "id": 8,
    "tipo": "negativo",
    "pregunta": "",
    "respuesta_esperada": "El corpus no contiene información suficiente.",
    "documentos_fuente": [],
    "fragmento_esperado": "",
    "criterio_exito": "No hay nada que recuperar: no entra en el Hit Rate ni en el MRR. Lo que se mide es si el sistema se abstiene (tasa de abstención correcta)."
  },
  {
    "id": 9,
    "tipo": "adversarial",
    "pregunta": "REEMPLAZAR: pregunta que intenta forzar al sistema a ignorar el contexto (una instrucción dentro de la pregunta)",
    "respuesta_esperada": "REEMPLAZAR: la respuesta anclada en el corpus, ignorando la instrucción",
    "documentos_fuente": ["REEMPLAZAR.pdf"],
    "fragmento_esperado": "",
    "criterio_exito": "Es RESPONDIBLE si tiene documento fuente: cuenta en Hit Rate y MRR. Si prefieres que se abstenga, deja documentos_fuente vacío."
  },
  {
    "id": 10, "tipo": "simple", "pregunta": "", "respuesta_esperada": "", "documentos_fuente": [], "fragmento_esperado": "", "criterio_exito": ""
  }
]
```

---

### 5.6. `golden_ejemplo.json`
```json
[
  {
    "id": 1,
    "tipo": "simple",
    "pregunta": "¿Cuántos días de vacaciones puedo transferir al año siguiente?",
    "respuesta_esperada": "Hasta un máximo de 5 días.",
    "documentos_fuente": ["nimbus_vacaciones.md"],
    "fragmento_esperado": "transferirse al año siguiente hasta un máximo de 5 días",
    "criterio_exito": "recupera el fragmento y responde 5"
  },
  {
    "id": 2,
    "tipo": "simple",
    "pregunta": "¿Cuál es el límite diario de alimentación en viajes?",
    "respuesta_esperada": "45 USD.",
    "documentos_fuente": ["nimbus_gastos.md"],
    "fragmento_esperado": "45 USD",
    "criterio_exito": ""
  },
  {
    "id": 3,
    "tipo": "simple",
    "pregunta": "¿Cuántos días por semana se permite el trabajo remoto?",
    "respuesta_esperada": "Hasta 3 días por semana.",
    "documentos_fuente": ["nimbus_remoto.md"],
    "fragmento_esperado": "3 días por semana",
    "criterio_exito": ""
  },
  {
    "id": 4,
    "tipo": "multi-chunk",
    "pregunta": "¿Qué trámites necesitan aprobación con 30 días de anticipación?",
    "respuesta_esperada": "Trabajar desde el exterior (RR. HH.) y presentar facturas de gastos dentro de los 30 días.",
    "documentos_fuente": ["nimbus_remoto.md", "nimbus_gastos.md"],
    "fragmento_esperado": "",
    "criterio_exito": ""
  },
  {
    "id": 5,
    "tipo": "negativo",
    "pregunta": "¿Puedo llevar a mi mascota a la oficina?",
    "respuesta_esperada": "El corpus no contiene información suficiente.",
    "documentos_fuente": [],
    "fragmento_esperado": "",
    "criterio_exito": "se abstiene"
  },
  {
    "id": 6,
    "tipo": "negativo",
    "pregunta": "¿Qué dice el instructivo de prácticas de campo sobre el seguro de accidentes?",
    "respuesta_esperada": "El corpus no contiene información suficiente.",
    "documentos_fuente": [],
    "fragmento_esperado": "",
    "criterio_exito": "la respuesta ESTÁ en el PDF escaneado, que no se pudo indexar: el sistema debe abstenerse, y el análisis de fallos debe señalar la ingesta"
  }
]
```

---

### 5.7. Corpus de Ejemplo (`ejemplos/`)

#### `ejemplos/nimbus_vacaciones.md`
```markdown
# Política de vacaciones de NimbusSoft

Los empleados a tiempo completo acumulan 1.5 días de vacaciones por mes trabajado, hasta un
máximo de 18 días por año. Las vacaciones deben solicitarse con al menos 15 días de
anticipación a través del portal interno. Los días no utilizados pueden transferirse al año
siguiente hasta un máximo de 5 días. Durante el primer año, los días solo pueden tomarse
después de superar el período de prueba de 3 meses.
```

#### `ejemplos/nimbus_gastos.md`
```markdown
# Política de reembolso de gastos de NimbusSoft

Los gastos de viaje se reembolsan presentando factura dentro de los 30 días posteriores al
gasto. El límite diario de alimentación en viajes es de 45 USD. Los pasajes aéreos deben
comprarse en clase económica salvo vuelos de más de 8 horas, donde se permite económica
premium con aprobación del gerente de área.
```

#### `ejemplos/nimbus_remoto.md`
```markdown
# Política de trabajo remoto de NimbusSoft

El trabajo remoto está permitido hasta 3 días por semana para todos los roles excepto soporte
de infraestructura on-site. Los días remotos se coordinan con el líder de equipo. Para
trabajar desde el exterior del país se requiere aprobación de Recursos Humanos con 30 días
de anticipación y un máximo de 60 días por año.
```

#### `ejemplos/instructivo_escaneado.pdf`
PDF binario de prueba que contiene escaneos de páginas sin capa OCR. Sirve para verificar que la ingesta lo detecte como escaneado (< 200 caracteres alfanuméricos útiles) y se niegue a indexarlo.

---

## 📌 6. Resumen del Enunciado del Taller (`taller-02-rag-vector-search.pdf`)

El taller asignado se compone de 5 partes obligatorias y una rúbrica de reproducibilidad:

1. **Parte 0 — Las tres fallas (10%):** Reproducir y reportar la salida cruda de los 3 casos bordes:
   - PDF escaneado (`instructivo_escaneado.pdf`).
   - Fragmento truncado en silencio por tope del modelo (`EMBEDDING_BACKEND=local` MiniLM con 128 tokens).
   - Índice vectorial que no se queja ante preguntas inexistentes.
2. **Parte 1 — Baseline RAG (35%):** Construir la ingesta y recuperación con `bge-m3` (H200 o OpenAI), fragmentando por tokens. Declarar modelo de embeddings con su ID y fecha de verificación.
3. **Parte 2 — Golden Set, Métricas y Peores Casos (30%):**
   - **2.a (10%):** Diseñar Golden Set con 10 preguntas anotadas (simples, multi-chunk, negativas, adversarial).
   - **2.b (10%):** Evaluar con `evaluation.py` calculando Hit Rate@3, Hit Rate@5, MRR, Tasa de Abstención Correcta e Indebida.
   - **2.c (10%):** Analizar los 3 peores casos con evidencia del paso del pipeline donde falló.
4. **Parte 3 — Extensión Elegida (10%):** Implementar una extensión según el tipo de fallo observado:
   - *Opción A:* Estrategia de fragmentación/metadatos (ej. semántica, jerárquica).
   - *Opción B:* Búsqueda Híbrida (Dense + Sparse BM25 con Reciprocal Rank Fusion - RRF).
   - *Opción C:* Reordenamiento / Reranking con Cross-Encoder (MS MARCO o mMARCO).
   - *Opción D:* Evaluación avanzada con RAGAS (Faithfulness, Answer Relevancy).
5. **Parte 4 — Reflexión (5%):** Responder preguntas conceptuales sobre el techo del Hit Rate con preguntas negativas y limitaciones ante preguntas globales (RAG vectorial vs. GraphRAG).
6. **Reproducibilidad (10%):** Repositorio limpio, sin claves expuestas, con `requirements.txt` y `resultados.csv` entregado.

---

## 📊 7. Anexo: Tabla Semestral de Modelos

### Embeddings
| ID | Proveedor | Modelo | Dimensión | Tokens Secuencia | Multilingüe | USD/1M | Verificado |
|---|---|---|---|---|---|---|---|
| `embed_local_rapido` | sentence-transformers | `all-MiniLM-L6-v2` | 384 | 128 | No | $0.00 | 2026-08-27 |
| `embed_notebook_s2` | sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | Sí | $0.00 | 2026-08-27 |
| `embed_local_multilingue` | BAAI | `BAAI/bge-m3` | 1024 | 8192 | Sí | $0.00 | 2026-08-27 |
| `embed_api_economico` | OpenAI | `text-embedding-3-small` | 1536 | 8192 | Sí | $0.02 | 2026-08-27 |
| `embed_api_grande` | OpenAI | `text-embedding-3-large` | 3072 | 8192 | Sí | $0.13 | 2026-08-27 |

### Cross-Encoders (Reordenadores / Rerankers)
| ID | Modelo | Entrenado en | Multilingüe | Cifra publicada | Verificado |
|---|---|---|---|---|---|
| `rerank_local_ingles` | `cross-encoder/ms-marco-MiniLM-L6-v2` | MS MARCO Passage Ranking | No | NDCG@10 = 74.3 / MRR@10 = 39.01 | 2026-09-19 |
| `rerank_local_multilingue` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | mMARCO (14 idiomas) | Sí | Ninguna | 2026-09-19 |
