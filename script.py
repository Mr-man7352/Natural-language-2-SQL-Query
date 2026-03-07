import tarfile
import requests
import json
from transformers import T5Tokenizer
from torch.utils.data import Dataset
import torch


def download_and_extract():
    """Download the WikiSQL dataset archive and extract it locally."""
    url = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open("data.tar.bz2", "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            print(f"\rDownloading: {downloaded}/{total} bytes ({100*downloaded//total}%)", end="")

    print("\nDone!")

    # Extract
    print("Extracting...")
    with tarfile.open("data.tar.bz2", "r:bz2") as tar:
        tar.extractall()


def convert_sql(sample, table):
  
  AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
  COND_OPS = ["=", ">", "<", "OP"]
  col_names= table['header']

  select_col= col_names[sample['sql']['sel']]

  agg= AGG_OPS[sample['sql']['agg']]

  if agg:
    select_part = f"{agg}({select_col})"
  else:
    select_part = select_col


  where_clause=[]

  for c in sample['sql']['conds']:
    column= col_names[c[0]]
    op= COND_OPS[c[1]]
    value= c[2]
    where_clause.append(f"{column}{op}'{value}'")

  where = " AND ".join(where_clause)

  query = f"SELECT {select_part} FROM table"

  if where:
    query += f" WHERE {where}"

  return query

# schema-aware input
def build_schema(table):
  return ", ".join([
  f"{col}({typ})"
  for col, typ in zip(table['header'], table['types'])
  ])

def create_training_example():
    training_pairs=[]

    with open("data/train.jsonl") as f_data, \
        open("data/train.tables.jsonl") as f_tables:

        tables={}

        for t in f_tables:
            table_obj= json.loads(t)
            tables[table_obj['id']]= table_obj

        for line in f_data:
            sample = json.loads(line)
            actual_table = tables[sample['table_id']]

            schema= build_schema(actual_table)

            sql = convert_sql(sample, actual_table)

            inp= f"Schema: {schema}, Question: {sample['question']}"

            training_pairs.append((inp,sql))

        training_pairs= training_pairs[:8000]
    return training_pairs


# Converting training pairs into tokenized data
class SQLDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, out = self.pairs[idx]
        tokens = self.tokenizer(
            inp,
            text_target=out,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["labels"]
        }

   
def main():

    # Entry point: trigger dataset download and extraction.
    # download_and_extract()

    with open("data/train.jsonl") as f:
        sample = json.loads(next(f))

    with open("data/train.tables.jsonl") as f:
        table = json.loads(next(f))

    print("QUESTION:\n", sample["question"])
    print("\nSQL LABEL:\n", sample["sql"])
    print("\nTABLE SCHEMA:\n", table)

    # convrt wikiSQL to SQL
    sql_query = convert_sql(sample, table)
    print("\nCONVERTED SQL QUERY:\n", sql_query)

    # create training examples
    training_pairs = create_training_example()
    print("\nSAMPLE TRAINING EXAMPLE:\n", len(training_pairs), training_pairs[0])

    # Tokenization and encoding for T5
    tokenizer= T5Tokenizer.from_pretrained("t5-small")
    tokenized_data= SQLDataset(training_pairs,tokenizer)
    print("\nSAMPLE TOKENIZED EXAMPLE:\n", tokenized_data[0])

    print("end of main")


if __name__ == "__main__":
    main()
