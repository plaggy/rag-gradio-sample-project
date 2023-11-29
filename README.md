# A template for a RAG system with Gradio UI
Deliberately stripped down to leave some room for experimenting

# Setting it up
- Clone https://github.com/huggingface/transformers to a local machine
- Use the **prep_scrips/markdown_to_text.py** script to extract raw text from markdown from transformers/docs/source/en/
- Break the resulting texts down into semantically meaningful pieces. Experiment with different chunking mechanisms to make sure the semantic meaning is captured.
- Use **prep_scrips/lancedb_setup.py** to embed and store chunks in a [lancedb](https://lancedb.github.io/lancedb/) instance. It also creates an index for fast ANN retrieval (not really needed for this exercise but necessary at scale). You'll need to put your own values into VECTOR_COLUMN_NAME, TEXT_COLUMN_NAME, DB_TABLE_NAME. If you are getting lancedb errors at the inference time try to drop the index because it might be not enough data to make it work.
- Move the database directory (.lancedb by default) to **gradio_app/**
- Use the template given in **gradio_app** to wrap everything into the [Gradio](https://www.gradio.app/docs/interface) app and run it on HF [spaces](https://huggingface.co/docs/hub/spaces-config-reference). Make sure to adjust VECTOR_COLUMN_NAME, TEXT_COLUMN_NAME, DB_TABLE_NAME according to your DB setup.
- In your space, set up secrets OPENAI_API_KEY and HUGGING_FACE_HUB_TOKEN to use OpenAI and open-source models correspondingly

- TODOs: 
  - Experiment with chunking, see how it affects the results. When deciding how to chunk it helps to think about what kind of chunks you'd like to see as context to your queries.
    - Deliverables: Demonstrate how retrieved documents differ with different chunking strategies and how it affects the output.
  - Try out different embedding models (EMB_MODEL_NAME). Good models to start with are **sentence-transformers/all-MiniLM-L6-v2** - lightweight, **thenlper/gte-large** - relatively heavy but more powerful.
    - Deliverables: Demonstrate how retrieved documents differ with different embedding models and how they affect the output. Provide an estimate of how the time to embed the chunks and DB ingestion time differs (happening in **prep_scrips/lancedb_setup.py**).
  - Add a re-ranker (cross-encoder) to the pipeline. Start with sentence-transformers pages on cross-encoders [1](https://www.sbert.net/examples/applications/cross-encoder/README.html) [2](https://www.sbert.net/examples/applications/retrieve_rerank/README.html), then pick a [pretrained cross-encoder](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html), e.g. **cross-encoder/ms-marco-MiniLM-L-12-v2**. Don't forget to increase the number of *retrieved* documents when using re-ranker. The number of documents used as context should stay the same.
    - Deliverables: Demonstrate how retrieved documents differ after adding a re-ranker and how it affects the output. Provide an estimate of how latency changes. 
  - Try another LLM (e.g. LLaMA-2-70b, falcon-180b-chat).
    - Deliverables: Demonstrate how LLMs affect the output and how latency changes with the model size.
  - Add more documents (e.g. diffusers, tokenizers, optimum, etc) to see how the system scales.
    - Deliverables: Demonstrate how latency changes, and how it differs with and without index (index is added in **prep_scrips/lancedb_setup.py**).
  - (Bonus) Use an LLM to quantitatively compare outputs of different variants of the system ([LLM as a Judge](https://huggingface.co/collections/andrewrreed/llm-as-a-judge-653fb861e361fd03c12d41e5))
    - Deliverables: Describe the experimental setup and evaluation results
