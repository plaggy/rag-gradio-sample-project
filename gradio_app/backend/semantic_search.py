import logging
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMB_MODEL_NAME = ""
DB_TABLE_NAME = ""

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
retriever = SentenceTransformer(EMB_MODEL_NAME)

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
table = db.open_table(DB_TABLE_NAME)
