import lancedb
import os
import gradio as gr
from sentence_transformers import SentenceTransformer


db = lancedb.connect(".lancedb")

TABLE = db.open_table(os.getenv("TABLE_NAME"))
VECTOR_COLUMN = os.getenv("VECTOR_COLUMN", "vector")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "text")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

retriever = SentenceTransformer(os.getenv("EMB_MODEL"))


def retrieve(query, k):
    query_vec = retriever.encode(query)
    try:
        documents = TABLE.search(query_vec, vector_column_name=VECTOR_COLUMN).limit(k).to_list()
        documents = [doc[TEXT_COLUMN] for doc in documents]

        return documents

    except Exception as e:
        raise gr.Error(str(e))
