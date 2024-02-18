import argparse
import os
import re

from tqdm import tqdm
from bs4 import BeautifulSoup
from markdown import markdown
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="input directory with markdown", type=str,
                        default="transformers/docs/source/en/")
    parser.add_argument("--output-dir", help="output directory to store raw texts", type=str,
                        default="docs")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    assert os.path.isdir(input_dir), "Input directory doesn't exist"

    files = input_dir.rglob("*")
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files):
        parent = file.parent.stem if file.parent.stem != input_dir.stem else ""
        if file.is_file():
            with open(file, encoding="utf-8") as f:
                md = f.read()

            text = markdown_to_text(md)

            with open(output_dir / f"{parent}_{file.stem}.txt", "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
