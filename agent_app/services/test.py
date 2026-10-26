from services import ( extract_pdf_pages, chunk_pages,)


PDF_PATH = "student_handbook.pdf"


pages = extract_pdf_pages(PDF_PATH)

chunks = chunk_pages(
    pages,
    chunk_size=450,
    overlap=75
)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")

if chunks:
    print(chunks[0])