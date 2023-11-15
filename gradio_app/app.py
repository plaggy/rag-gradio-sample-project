"""
Credit to Derek Thomas, derek@huggingface.co
"""

import subprocess

subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

import logging
from pathlib import Path
from time import perf_counter

import gradio as gr
from jinja2 import Environment, FileSystemLoader

from backend.query_llm import generate_hf, generate_openai
from backend.semantic_search import table, retriever

VECTOR_COLUMN_NAME = ""
TEXT_COLUMN_NAME = ""

proj_dir = Path(__file__).parent
# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader(proj_dir / 'templates'))

# Load the templates directly from the environment
template = env.get_template('template.j2')
template_html = env.get_template('template_html.j2')

# Examples
examples = ['What is the capital of China?',
            'Why is the sky blue?',
            'Who won the mens world cup in 2014?', ]


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, api_kind):
    top_k_rank = 4
    query = history[-1][0]

    if not query:
         gr.Warning("Please submit a non-empty string as a prompt")
         raise ValueError("Empty string was submitted")

    logger.warning('Retrieving documents...')
    # Retrieve documents relevant to query
    document_start = perf_counter()

    query_vec = retriever.encode(query)
    documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN_NAME).limit(top_k_rank).to_list()
    documents = [doc[TEXT_COLUMN_NAME] for doc in documents]

    document_time = perf_counter() - document_start
    logger.warning(f'Finished Retrieving documents in {round(document_time, 2)} seconds...')

    # Create Prompt
    prompt = template.render(documents=documents, query=query)
    prompt_html = template_html.render(documents=documents, query=query)

    if api_kind == "HuggingFace":
         generate_fn = generate_hf
    elif api_kind == "OpenAI":
         generate_fn = generate_openai
    elif api_kind is None:
         gr.Warning("API name was not provided")
         raise ValueError("API name was not provided")
    else:
         gr.Warning(f"API {api_kind} is not supported")
         raise ValueError(f"API {api_kind} is not supported")

    history[-1][1] = ""
    for character in generate_fn(prompt, history[:-1]):
        history[-1][1] = character
        yield history, prompt_html


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            avatar_images=('https://aui.atlassian.com/aui/8.8/docs/images/avatar-person.svg',
                           'https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg'),
            bubble_full_width=False,
            show_copy_button=True,
            show_share_button=True,
            )

    with gr.Row():
        txt = gr.Textbox(
                scale=3,
                show_label=False,
                placeholder="Enter text and press enter",
                container=False,
                )
        txt_btn = gr.Button(value="Submit text", scale=1)

    api_kind = gr.Radio(choices=["HuggingFace", "OpenAI"], value="HuggingFace")

    prompt_html = gr.HTML()
    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [chatbot, api_kind], [chatbot, prompt_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [chatbot, api_kind], [chatbot, prompt_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Examples
    gr.Examples(examples, txt)

demo.queue()
demo.launch(debug=True)
