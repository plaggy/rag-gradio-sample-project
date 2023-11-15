from bs4 import BeautifulSoup
from markdown import markdown
import os
import re
from pathlib import Path


DIR_TO_SCRAPE = "transformers/docs/source/en/"
OUTPUT_DIR = str(Path().resolve() / "docs_dump")


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    html = re.sub(r'<!--((.|\n)*)-->', '', html)
    html = re.sub('<code>bash', '<code>', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    text = re.sub('```(py|diff|python)', '', text)
    text = re.sub('```\n', '\n', text)
    text = re.sub('-         .*', '', text)
    text = text.replace('...', '')
    text = re.sub('\n(\n)+', '\n\n', text)

    return text


dir_to_scrape = Path(DIR_TO_SCRAPE)
files = list(dir_to_scrape.rglob("*"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in files:
    parent = file.parent.stem if file.parent.stem != dir_to_scrape.stem else ""
    if file.is_file():
        with open(file) as f:
            md = f.read()

        text = markdown_to_text(md)

        with open(os.path.join(OUTPUT_DIR, f"{parent}_{file.stem}.txt"), "w") as f:
            f.write(text)

