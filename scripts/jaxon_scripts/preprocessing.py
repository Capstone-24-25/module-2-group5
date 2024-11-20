import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import regex


# Function to parse HTML and clean text
def parse_fn(html):
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    text_list = [p.get_text(strip=True) for p in paragraphs]
    text = " ".join(text_list)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # Remove all apostrophes
    text = text.replace("'", "")
    # Replace newline characters and 'nbsp' with space
    text = text.replace("\n", " ").replace("nbsp", " ")
    # Remove punctuation, digits, and symbols
    text = regex.sub(r"[\p{P}\p{S}\p{N}]", " ", text)
    # Insert space between lowercase and uppercase letters
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Convert to lowercase
    text = text.lower()
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Function to apply to claims data
def parse_data(df):
    # Filter rows where 'text_tmp' contains '<!'
    df_filtered = df[df["text_tmp"].str.contains("<!", na=False)].copy()
    # Apply 'parse_fn' to 'text_tmp' column
    df_filtered["text_clean"] = df_filtered["text_tmp"].apply(parse_fn)
    return df_filtered


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


# if __name__ == "__main__":
#     pass
