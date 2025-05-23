from vector_db_manager import create_new_vectorstore_from_pdfs

fichas = [
    "./data/tahoe.pdf",
    "./data/sail-my25.pdf",
    "./data/onix-turbo-my25.pdf"
]

create_new_vectorstore_from_pdfs(fichas)
print("✅ Base vectorial creada con fichas técnicas.")