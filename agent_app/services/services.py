import fitz
from django.db import transaction
from agent_app.models import HandbookModel, DocumentChunkModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# def extract_pdf_pages(pdf_path):
#     """
#     Extract text from a PDF while preserving
#     the physical PDF page number.
#     """

#     extracted_pages = []

#     with fitz.open(pdf_path) as document:
#         for page_number, page in enumerate(
#             document,
#             start=1
#         ):
#             text = page.get_text("text").strip()

#             extracted_pages.append({
#                 "page_number": page_number,
#                 "text": text,
#             })

#     return extracted_pages

def extract_pdf_pages(pdf_path):
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_index in range(len(pdf)):
            current_page = pdf.load_page(page_index)
            text = current_page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_index + 1,
                        "source": str(pdf_path),
                    },
                )
            )

    return pages

def chunk_pages():
    pass

# def chunk_pages(
#     pages,
#     chunk_size=450,
#     overlap=75
# ):
#     """
#     Split extracted page text into overlapping
#     chunks while preserving source page numbers.

#     chunk_size and overlap are measured in words,
#     not model tokens.
#     """

#     if chunk_size <= 0:
#         raise ValueError(
#             "chunk_size must be positive"
#         )

#     if overlap < 0 or overlap >= chunk_size:
#         raise ValueError(
#             "overlap must be non-negative "
#             "and smaller than chunk_size"
#         )

#     chunks = []

#     for page in pages:
#         page_number = page["page_number"]
#         text = page["text"]

#         if not text:
#             continue

#         words = text.split()

#         start = 0

#         while start < len(words):
#             end = min(
#                 start + chunk_size,
#                 len(words)
#             )

#             chunk_text = " ".join(
#                 words[start:end]
#             )

#             chunks.append({
#                 "content": chunk_text,
#                 "page_start": page_number,
#                 "page_end": page_number,
#             })

#             if end == len(words):
#                 break

#             start = end - overlap

#     return chunks

def split_pages_into_chunks(pages):
    # Stage 1: Split by characters
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0,
    )

    character_chunks = character_splitter.split_documents(pages)

    # Stage 2: Split each character chunk by tokens
    token_splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk=256,
        chunk_overlap=0,
    )

    token_chunks = []

    for chunk in character_chunks:
        token_chunks.extend(
            token_splitter.split_documents([chunk])
        )

    return character_chunks, token_chunks



def process_handbook(handbook_id):
    handbook = HandbookModel.objects.get(id=handbook_id)
    handbook.status = "processing"
    handbook.error_message = ""
    handbook.save( update_fields=["status", "error_message", "updated_at"] )
    try:
        pages = extract_pdf_pages( handbook.file.path )
        chunks = chunk_pages(pages)

        with transaction.atomic():
            DocumentChunkModel.objects.filter(handbook=handbook).delete()
            DocumentChunkModel.objects.bulk_create([
                DocumentChunkModel(
                    handbook=handbook,
                    content=chunk["content"],
                    page_start=chunk["page_start"],
                    page_end=chunk["page_end"],
                    chunk_index=index,
                )
                for index, chunk in enumerate(chunks)
            ])

            handbook.total_pages = len(pages)
            handbook.status = "completed"
            handbook.save(
                update_fields=[
                    "total_pages",
                    "status",
                    "updated_at",
                ]
            )

        return {
            "handbook_id": handbook.id,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "status": "completed",
        }

    except Exception as exc:
        handbook.status = "failed"
        handbook.error_message = str(exc)
        handbook.save(
            update_fields=[
                "status",
                "error_message",
                "updated_at",
            ]
        )
        raise