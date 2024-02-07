import argparse
import lancedb
import torch
import pyarrow as pa
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np
import logging

from transformers import AutoConfig
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-model", help="embedding model name on HF hub", type=str)
    parser.add_argument("--table", help="table name in DB", type=str)
    parser.add_argument("--input-dir", help="input directory with documents to ingest", type=str)
    parser.add_argument("--vec-column", help="vector column name in the table", type=str, default="vector")
    parser.add_argument("--text-column", help="text column name in the table", type=str, default="text")
    parser.add_argument("--db-loc", help="database location", type=str,
                        default=str(Path().resolve() / "gradio_app" / ".lancedb"))
    parser.add_argument("--batch-size", help="batch size for embedding model", type=int, default=32)
    parser.add_argument("--num-partitions", help="number of partitions for index", type=int, default=256)
    parser.add_argument("--num-sub-vectors", help="number of sub-vectors for index", type=int, default=96)

    args = parser.parse_args()

    emb_config = AutoConfig.from_pretrained(args.emb_model)
    emb_dimension = emb_config.hidden_size

    assert emb_dimension % args.num_sub_vectors == 0, \
        "Embedding size must be divisible by the num of sub vectors"

    model = SentenceTransformer(args.emb_model)
    model.eval()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"using {str(device)} device")

    db = lancedb.connect(args.db_loc)

    schema = pa.schema(
      [
          pa.field(args.vec_column, pa.list_(pa.float32(), emb_dimension)),
          pa.field(args.text_column, pa.string())
      ]
    )
    tbl = db.create_table(args.table, schema=schema, mode="overwrite")

    input_dir = Path(args.input_dir)
    files = list(input_dir.rglob("*"))

    sentences = []
    for file in files:
        with open(file) as f:
            sentences.append(f.read())

    for i in tqdm.tqdm(range(0, int(np.ceil(len(sentences) / args.batch_size)))):
        try:
            batch = [sent for sent in sentences[i * args.batch_size:(i + 1) * args.batch_size] if len(sent) > 0]
            encoded = model.encode(batch, normalize_embeddings=True, device=device)
            encoded = [list(vec) for vec in encoded]

            df = pd.DataFrame({
                args.vec_column: encoded,
                args.text_column: batch
            })

            tbl.add(df)
        except:
            logger.info(f"batch {i} was skipped")

    '''
    create ivf-pd index https://lancedb.github.io/lancedb/ann_indexes/
    with the size of the transformer docs, index is not really needed
    but we'll do it for demonstrational purposes
    '''
    tbl.create_index(
        num_partitions=args.num_partitions,
        num_sub_vectors=args.num_sub_vectors,
        vector_column_name=args.vec_column
    )


if __name__ == "__main__":
    main()