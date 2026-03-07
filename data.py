import tarfile
import requests
import json



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

    


if __name__ == "__main__":
    main()
