import pandas as pd
import numpy as np
import spacy
import re
from bs4 import BeautifulSoup


def extract_headers(soup):
    headers = []
    for header_tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for header in soup.find_all(header_tag):
            headers.append(header.get_text(strip=True))
    headers = " ".join(headers)
    return headers


def extract_paragraphs(soup):
    contents = []
    for paragraph in soup.find_all("p"):
        contents.append(paragraph.get_text(strip=True))
    contents = " ".join(contents)
    return contents


if __name__ == "__main__":
    pass
