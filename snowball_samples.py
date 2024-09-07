from vars.internal_ids import get_internal_paper_id, get_internal_author_id, increment_internal_author_id, increment_internal_paper_id

import argparse
import os

parser = argparse.ArgumentParser(description="Load API keys as environment variables.")
parser.add_argument('--SEMANTIC_KEY', type=str, help='Semantic API key')
parser.add_argument('--GPT_KEY', type=str, help='GPT API key')

args = parser.parse_args()

if args.SEMANTIC_KEY:
    os.environ['SEMANTIC_KEY'] = args.SEMANTIC_KEY
    print("SEMANTIC_KEY has been set.")
else:
    raise ValueError("SEMANTIC_KEY not provided!")

if args.GPT_KEY:
    os.environ['GPT_KEY'] = args.GPT_KEY
    print("GPT_KEY has been set.")
else:
    raise ValueError("GPT_KEY not provided!")

SEMANTIC_KEY = os.getenv('SEMANTIC_KEY')
GPT_KEY = os.getenv('GPT_KEY')

from data_utils import read_parquet_file, write_parquet_file

import requests
import json

import time

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from collections import deque

from openai import OpenAI

import time

from copy import deepcopy

from tqdm import tqdm


def get_author_info_ss(api_key, author_id):
    query = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"

    author_fields = ["authorId", "name", "externalIds", "citationCount", "hIndex"]
    paper_fields = ["papers"]

    fields = ",".join(author_fields + paper_fields)

    time.sleep(15)
    r = requests.get(
        query,
        params = {"fields": f"{fields}"},
        headers={"x-api-key": api_key}
    )
    if r.status_code != 200:
        print(f"API ERROR: {r.status_code}\n{r.reason}\n\n")
        time.sleep(30)

        r = requests.get(
            query,
            params = {"fields": f"{fields}"},
            headers={"x-api-key": api_key}
        )

    r = r.json()

    author_info = {
        "author_id_semanticscholar": r["authorId"] if "authorId" in r else None,
        "ss_name": r["name"] if "name" in r else None,
        "DBLP_name": r["externalIds"]["DBLP"] if 'externalIds' in r and 'DBLP' in r['externalIds'] else None,
        "citations": r["citationCount"] if "citationCount" in r else None,
        "h_index": r["hIndex"] if "hIndex" in r else None
    }

    papers_endpoint_query = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
    fields = ["externalIds", "authors"]
    fields = ",".join(fields)

    time.sleep(15)
    r_paper = requests.get(
        papers_endpoint_query,
        params = {"fields": f"{fields}"},
        headers={"x-api-key": api_key}
    )
    if r_paper.status_code != 200:
        print(f"API ERROR: {r.status_code}\n{r.reason}\n\n")
        time.sleep(30)

        r_paper = requests.get(
        papers_endpoint_query,
        params = {"fields": f"{fields}"},
        headers={"x-api-key": api_key}
        )

    r_paper = r_paper.json()

    paper_info_author = r["papers"] if "papers" in r else []

    all_papers = []
    for paper in r_paper['data']:
        paper = {
            "semantic_scholar_id": paper["paperId"] if "paperId" in paper else None,
            'doi': paper['externalIds']['DOI'] if 'externalIds' in paper and 'DOI' in paper['externalIds'] else None,
            'authors': [{'author_id': author['authorId'] if "authorId" in author else None, 
                         'name': author['name'] if "name" in author else None} for author in paper['authors']]
        }

        all_papers.append(paper)

    return author_info, paper_info_author, all_papers


def get_paper_ssinfo(api_key, paper_id):
    query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

    identifying_fields = ["externalIds", "publicationDate"]

    paper_fields = ["title", "openAccessPdf"]
    reference_fields = ["citationCount"]

    author_fields = ["authors"]

    fields = identifying_fields + paper_fields + reference_fields + author_fields
    fields = ",".join(fields)

    time.sleep(15)
    r = requests.get(
        query,
        params = {"fields": f"{fields}"},
        headers={"x-api-key": api_key}
    )
    if r.status_code != 200:
        print(f"API ERROR: {r.status_code}\n{r.reason}\n\n")
        time.sleep(30)
        
        r = requests.get(
        query,
        params = {"fields": f"{fields}"},
        headers={"x-api-key": api_key}
        )

    r = r.json()

    return {
        'paper_id_semanticscholar': r['paperId'] if "paperId" in r else None,
        'external_ids': r['externalIds'] if "externalIds" in r else None,
        'doi': r['externalIds']['DOI'] if 'externalIds' in r and 'DOI' in r['externalIds'] else None,
        'title': r['title'] if "title" in r else None,            
        'citations': r['citationCount'] if "citationCount" in r else None,
        'open_access_url': r['openAccessPdf']['url'] if r['openAccessPdf'] else None,
        'publication_date': r['publicationDate'] if "publicationDate" in r else None,
        'authors': [{'author_id': author['authorId'] if "authorId" in author else None, 
                         'name': author['name'] if "name" in author else None} for author in r['authors'] if "authors" in r]
        }


def get_abstract(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        abstract_inverted_index = data.get("abstract_inverted_index", None)
    else:
        return ""

    if abstract_inverted_index == None:
        return ""
    
    max_index = max([max(indices) for indices in abstract_inverted_index.values()])
    reconstructed_text = [""] * (max_index + 1)
    
    for word, indices in abstract_inverted_index.items():
        for index in indices:
            reconstructed_text[index] = word
        
    return " ".join(reconstructed_text)


def embed_title_abstract(text, key):
    client = OpenAI(api_key=key)

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    response = response.data[0].embedding
  
    return response 


def cycle():
    to_mine_df = read_parquet_file("data/to_mine.parquet")
    author_info_df = read_parquet_file("data/author_info.parquet")
    paper_info_df = read_parquet_file("data/paper_info.parquet")

    to_mine_df = to_mine_df.sort_values(by='relevance', ascending=False)
    top_item = to_mine_df.iloc[0]

    print(top_item)

    if top_item["semantic_scholar_id"] in paper_info_df["paper_id_semanticscholar"].values:
        to_mine_df = to_mine_df[to_mine_df['semantic_scholar_id'] != top_item["semantic_scholar_id"]]
        write_parquet_file(to_mine_df, "data/to_mine.parquet")
        return True
    
    authors = top_item["authors"]
    authors = deepcopy(authors)
        

    paper_data = get_paper_ssinfo(SEMANTIC_KEY, top_item["semantic_scholar_id"])

    internal_paper_id = get_internal_paper_id()
    doi = paper_data["doi"]

    title = paper_data["title"]
    abstract = get_abstract(doi)

    text = title + abstract
    embedding = embed_title_abstract(text, GPT_KEY)

    paper_data["internal_id_paper"] = str(internal_paper_id).zfill(6)
    paper_data["embedding"] = embedding 

    paper_data = pd.DataFrame([paper_data])
    paper_info_df = pd.concat([paper_info_df, paper_data], ignore_index=True, sort=False)
    increment_internal_paper_id()

    for author in tqdm(authors):

        author_id_internal = get_internal_author_id()
        semantic_scholar_author_id = author["author_id"]

        if semantic_scholar_author_id in author_info_df["author_id_semanticscholar"].values:
            continue

        author_info, paper_info_author, all_papers = get_author_info_ss(SEMANTIC_KEY, semantic_scholar_author_id)
        paper_info = [item["paperId"] for item in paper_info_author]
        
        author_info["internal_id_author"] = str(author_id_internal).zfill(6)
        author_info["papers"] = paper_info

        author_data = pd.DataFrame([author_info])
        author_info_df = pd.concat([author_info_df, author_data], ignore_index=True, sort=False)
        increment_internal_author_id()
        
        for paper_potential in all_papers:
            authors_potential = []

            if paper_potential["semantic_scholar_id"] in to_mine_df["semantic_scholar_id"].values:
                pass

            authors_potential = [author_potential["author_id"] for author_potential in paper_potential["authors"]]
            new_authors_count = 0

            for author_potential in authors_potential:
                if author_potential not in author_info_df["author_id_semanticscholar"].values:
                    new_authors_count += 1
            
            relevance_score = new_authors_count * 1

            if relevance_score > 0 and relevance_score <= 20:
                paper_potential["relevance"] = relevance_score
                paper_potential = pd.DataFrame([paper_potential])
                to_mine_df = pd.concat([to_mine_df, paper_potential], ignore_index=True, sort=False)

            else:
                continue


    to_mine_df = to_mine_df[to_mine_df['semantic_scholar_id'] != top_item["semantic_scholar_id"]]

    write_parquet_file(author_info_df, "data/author_info.parquet")
    write_parquet_file(paper_info_df, "data/paper_info.parquet")
    write_parquet_file(to_mine_df, "data/to_mine.parquet")


cycle()