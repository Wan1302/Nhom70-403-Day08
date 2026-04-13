"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_OPENAI_CLIENT = None

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db_runtime"
FALLBACK_INDEX_PATH = CHROMA_DB_DIR / "rag_lab_fallback.json"

# TODO Sprint 1: Điều chỉnh chunk size và overlap theo quyết định của nhóm
# Gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access

    TODO Sprint 1:
    - Extract metadata từ dòng đầu file (Source, Department, Effective Date, Access)
    - Bỏ các dòng header metadata khỏi nội dung chính
    - Normalize khoảng trắng, xóa ký tự rác

    Gợi ý: dùng regex để parse dòng "Key: Value" ở đầu file.
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    metadata_pattern = re.compile(r"^(Source|Department|Effective Date|Access):\s*(.+)$", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if not header_done:
            # TODO: Parse metadata từ các dòng "Key: Value"
            # Ví dụ: "Source: policy/refund-v4.pdf" → metadata["source"] = "policy/refund-v4.pdf"
            match = metadata_pattern.match(stripped)
            if match:
                key, value = match.group(1).lower(), match.group(2).strip()
                if key == "source":
                    metadata["source"] = value
                elif key == "department":
                    metadata["department"] = value
                elif key == "effective date":
                    metadata["effective_date"] = value
                elif key == "access":
                    metadata["access"] = value
            elif stripped.startswith("==="):
                # Gặp section heading đầu tiên → kết thúc header
                header_done = True
                content_lines.append(stripped)
            elif stripped == "" or stripped.isupper():
                # Dòng tên tài liệu (toàn chữ hoa) hoặc dòng trống
                continue
            else:
                header_done = True
                content_lines.append(stripped)
        else:
            content_lines.append(line.rstrip())

    cleaned_text = "\n".join(content_lines)

    # TODO: Thêm bước normalize text nếu cần
    # Gợi ý: bỏ ký tự đặc biệt thừa, chuẩn hóa dấu câu
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # max 2 dòng trống liên tiếp
    cleaned_text = cleaned_text.strip()

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó

    TODO Sprint 1:
    1. Split theo heading "=== Section ... ===" hoặc "=== Phần ... ===" trước
    2. Nếu section quá dài (> CHUNK_SIZE * 4 ký tự), split tiếp theo paragraph
    3. Thêm overlap: lấy đoạn cuối của chunk trước vào đầu chunk tiếp theo
    4. Mỗi chunk PHẢI giữ metadata đầy đủ từ tài liệu gốc

    Gợi ý: Ưu tiên cắt tại ranh giới tự nhiên (section, paragraph)
    thay vì cắt theo token count cứng.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # TODO: Implement chunking theo section heading
    # Bước 1: Split theo heading pattern "=== ... ==="
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks với overlap.

    TODO Sprint 1:
    Hiện tại dùng split đơn giản theo ký tự.
    Cải thiện: split theo paragraph (\n\n) trước, rồi mới ghép đến khi đủ size.
    """
    if len(text) <= chunk_chars:
        # Toàn bộ section vừa một chunk
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    # TODO: Implement split theo paragraph với overlap
    # Gợi ý:
    # paragraphs = text.split("\n\n")
    # Ghép paragraphs lại cho đến khi gần đủ chunk_chars
    # Lấy overlap từ đoạn cuối chunk trước
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_parts = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current_parts and current_len + paragraph_len + 2 > chunk_chars:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append({
                "text": chunk_text,
                "metadata": {**base_metadata, "section": section},
            })

            overlap = chunk_text[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current_parts = [overlap, paragraph] if overlap else [paragraph]
            current_len = sum(len(part) for part in current_parts) + 2 * (len(current_parts) - 1)
        elif paragraph_len > chunk_chars:
            if current_parts:
                chunks.append({
                    "text": "\n\n".join(current_parts).strip(),
                    "metadata": {**base_metadata, "section": section},
                })
                current_parts = []
                current_len = 0

            start = 0
            while start < paragraph_len:
                end = min(start + chunk_chars, paragraph_len)

                # TODO: Tìm ranh giới tự nhiên gần nhất (dấu xuống dòng, dấu chấm)
                # thay vì cắt giữa câu
                if end < paragraph_len:
                    boundary = max(paragraph.rfind(". ", start, end), paragraph.rfind("\n", start, end))
                    if boundary > start + chunk_chars * 0.6:
                        end = boundary + 1

                chunk_text = paragraph[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {**base_metadata, "section": section},
                    })
                if end >= paragraph_len:
                    break
                start = max(end - overlap_chars, start + 1)
        else:
            current_parts.append(paragraph)
            current_len += paragraph_len + (2 if current_parts else 0)

    if current_parts:
        chunks.append({
            "text": "\n\n".join(part for part in current_parts if part).strip(),
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.

    TODO Sprint 1:
    Chọn một trong hai:

    Option A — OpenAI Embeddings (cần OPENAI_API_KEY):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    Option B — Sentence Transformers (chạy local, không cần API key):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model.encode(text).tolist()
    """
    global _OPENAI_CLIENT

    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if provider != "openai":
        raise ValueError(
            f"EMBEDDING_PROVIDER={provider!r} không được hỗ trợ trong cấu hình hiện tại. "
            "Hãy đặt EMBEDDING_PROVIDER=openai trong .env."
        )

    if not openai_key or openai_key.startswith("sk-..."):
        raise RuntimeError("Thiếu OPENAI_API_KEY hợp lệ trong .env.")

    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=openai_key)

    response = _OPENAI_CLIENT.embeddings.create(
        input=text,
        model=embedding_model,
    )
    return response.data[0].embedding


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.

    TODO Sprint 1:
    1. Cài thư viện: pip install chromadb
    2. Khởi tạo ChromaDB client và collection
    3. Với mỗi file trong docs_dir:
       a. Đọc nội dung
       b. Gọi preprocess_document()
       c. Gọi chunk_document()
       d. Với mỗi chunk: gọi get_embedding() và upsert vào ChromaDB
    4. In số lượng chunk đã index

    Gợi ý khởi tạo ChromaDB:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_or_create_collection(
            name="rag_lab",
            metadata={"hnsw:space": "cosine"}
        )
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Khởi tạo ChromaDB
    # client = chromadb.PersistentClient(path=str(db_dir))
    # collection = client.get_or_create_collection(...)
    collection = None
    fallback_records = []
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            client.delete_collection("rag_lab")
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name="rag_lab",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        print(f"ChromaDB không mở được ({exc}). Dùng JSON vector-store fallback.")

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        # TODO: Gọi preprocess_document
        # doc = preprocess_document(raw_text, str(filepath))

        # TODO: Gọi chunk_document
        # chunks = chunk_document(doc)

        # TODO: Embed và lưu từng chunk vào ChromaDB
        # for i, chunk in enumerate(chunks):
        #     chunk_id = f"{filepath.stem}_{i}"
        #     embedding = get_embedding(chunk["text"])
        #     collection.upsert(
        #         ids=[chunk_id],
        #         embeddings=[embedding],
        #         documents=[chunk["text"]],
        #         metadatas=[chunk["metadata"]],
        #     )
        # total_chunks += len(chunks)

        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])
            if collection is not None:
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]],
                )
            else:
                fallback_records.append({
                    "id": chunk_id,
                    "embedding": embedding,
                    "document": chunk["text"],
                    "metadata": chunk["metadata"],
                })

        print(f"    → {len(chunks)} chunks indexed")
        total_chunks += len(chunks)

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")
    if collection is None:
        FALLBACK_INDEX_PATH.write_text(json.dumps(fallback_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Index đã được lưu vào JSON fallback: {FALLBACK_INDEX_PATH}")
    else:
        print("Index đã được lưu vào ChromaDB.")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.

    TODO Sprint 1:
    Implement sau khi hoàn thành build_index().
    Kiểm tra:
    - Chunk có giữ đủ metadata không? (source, section, effective_date)
    - Chunk có bị cắt giữa điều khoản không?
    - Metadata effective_date có đúng không?
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:120]}...")
            print()
    except Exception as e:
        if FALLBACK_INDEX_PATH.exists():
            records = json.loads(FALLBACK_INDEX_PATH.read_text(encoding="utf-8"))
            print(f"\n=== Top {n} chunks trong JSON fallback ===\n")
            for i, record in enumerate(records[:n]):
                meta = record.get("metadata", {})
                doc = record.get("document", "")
                print(f"[Chunk {i+1}]")
                print(f"  Source: {meta.get('source', 'N/A')}")
                print(f"  Section: {meta.get('section', 'N/A')}")
                print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
                print(f"  Text preview: {doc[:120]}...")
                print()
        else:
            print(f"Lỗi khi đọc index: {e}")
            print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.

    Checklist Sprint 1:
    - Mọi chunk đều có source?
    - Có bao nhiêu chunk từ mỗi department?
    - Chunk nào thiếu effective_date?

    TODO: Implement sau khi build_index() hoàn thành.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        print(f"\nTổng chunks: {len(results['metadatas'])}")

        # TODO: Phân tích metadata
        # Đếm theo department, kiểm tra effective_date missing, v.v.
        departments = {}
        missing_date = 0
        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date: {missing_date}")

    except Exception as e:
        if FALLBACK_INDEX_PATH.exists():
            records = json.loads(FALLBACK_INDEX_PATH.read_text(encoding="utf-8"))
            print(f"\nTổng chunks: {len(records)}")

            # TODO: Phân tích metadata
            departments = {}
            missing_date = 0
            for record in records:
                meta = record.get("metadata", {})
                dept = meta.get("department", "unknown")
                departments[dept] = departments.get(dept, 0) + 1
                if meta.get("effective_date") in ("unknown", "", None):
                    missing_date += 1

            print("Phân bố theo department:")
            for dept, count in departments.items():
                print(f"  {dept}: {count} chunks")
            print(f"Chunks thiếu effective_date: {missing_date}")
        else:
            print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:  # Test với 1 file đầu
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index (yêu cầu implement get_embedding)
    print("\n--- Build Full Index ---")
    print("Đang build index đầy đủ...")
    # Uncomment dòng dưới sau khi implement get_embedding():
    build_index()

    # Bước 4: Kiểm tra index
    # Uncomment sau khi build_index() thành công:
    list_chunks()
    inspect_metadata_coverage()

    print("\nSprint 1 setup hoàn thành!")
    print("Việc cần làm:")
    print("  1. Implement get_embedding() - chọn OpenAI hoặc Sentence Transformers")
    print("  2. Implement phần TODO trong build_index()")
    print("  3. Chạy build_index() và kiểm tra với list_chunks()")
    print("  4. Nếu chunking chưa tốt: cải thiện _split_by_size() để split theo paragraph")
