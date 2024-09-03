import os
if not os.path.isdir("./data"): os.mkdir("./data")

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def initialize_schema():
    schema_tomine = pa.schema([
        ('semantic_scholar_id', pa.string()),
        ('doi', pa.string()),
        ('authors', pa.list_(pa.struct([
            ('author_id', pa.string()),
            ('name', pa.string())
        ]))),
        ('relevance', pa.int64())
    ])

    schema_paperinfo = pa.schema([
        ('internal_id_paper', pa.string()),
        ('paper_id_semanticscholar', pa.string()),
        ('external_ids', pa.map_(pa.string(), pa.string())),
        ('doi', pa.string()),
        ('embedding', pa.list_(pa.float32())),
        ('title', pa.string()),
        ('abstract', pa.string()),
        ('citations', pa.int64()),
        ('open_access_url', pa.string()),
        ('publication_date', pa.string()),
        ('authors', pa.list_(pa.struct([
            ('author_id', pa.string()),
            ('name', pa.string())
        ])))
    ])

    schema_authorinfo = pa.schema([
        ('internal_id_author', pa.string()),
        ('author_id_semanticscholar', pa.string()),
        ('ss_name', pa.string()),
        ('DBLP_name', pa.string()),
        ('citations', pa.int64()),
        ('h_index', pa.int64()),
        ('papers', pa.list_(pa.struct([
            ('paper_id_semanticscholar', pa.string()),
        ])))
    ])

    pq.write_table(pa.Table.from_pandas(pd.DataFrame(columns=schema_tomine.names), schema=schema_tomine), "./data/to_mine.parquet")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(columns=schema_paperinfo.names), schema=schema_paperinfo), "./data/paper_info.parquet")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(columns=schema_authorinfo.names), schema=schema_authorinfo), "./data/author_info.parquet")
    
    return schema_tomine, schema_paperinfo, schema_authorinfo

schema_tomine, schema_paperinfo, schema_authorinfo = initialize_schema()

data = {
    'semantic_scholar_id': ['6364fdaa0a0eccd823a779fcdd489173f938e91a'],
    'doi': ['10.1007/978-3-319-24574-4_28'],
    'authors': [[{'author_id': '1737326', 'name': 'O. Ronneberger'},
                 {'author_id': '152702479', 'name': 'P. Fischer'},
                 {'author_id': '1710872', 'name': 'T. Brox'}]],
    'relevance': [0]
}

df = pd.DataFrame(data)
df.to_parquet("./data/to_mine.parquet", engine='pyarrow', index=False)