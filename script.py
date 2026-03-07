import tarfile
import requests
import json
import time
from transformers import T5Tokenizer
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
import torch
import sqlparse

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

class TimingCallback(TrainerCallback):
    def __init__(self):
        self.train_start = None
        self.epoch_start = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        total_steps = state.max_steps
        print(f"\n{'='*55}")
        print(f"  Training started — Total steps: {total_steps}")
        print(f"{'='*55}\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        epoch = int(state.epoch or 0) + 1
        print(f"\n── Epoch {epoch} started ──")

    def on_step_end(self, args, state, control, **kwargs):
        # Record time every 10 steps and print a progress line
        if state.global_step % 10 == 0 and state.global_step > 0:
            elapsed = time.time() - self.train_start
            steps_done = state.global_step
            total_steps = state.max_steps
            steps_left = total_steps - steps_done
            avg_sec_per_step = elapsed / steps_done
            eta_sec = avg_sec_per_step * steps_left

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str     = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
            pct         = 100 * steps_done / total_steps

            print(
                f"  Step {steps_done:>4}/{total_steps} "
                f"({pct:5.1f}%) | "
                f"Elapsed: {elapsed_str} | "
                f"ETA: {eta_str}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_elapsed = time.time() - self.epoch_start
        total_elapsed = time.time() - self.train_start
        epoch = int(state.epoch or 0)
        print(f"\n── Epoch {epoch} done in {epoch_elapsed:.1f}s "
              f"(total: {total_elapsed:.1f}s) ──\n")

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.train_start
        total_str = time.strftime("%H:%M:%S", time.gmtime(total))
        print(f"\n{'='*55}")
        print(f"  Training complete!  Total time: {total_str}")
        print(f"{'='*55}\n")

def is_valid_sql(sql):
    return bool(sqlparse.parse(sql))

def generate_sql(tokenizer,model,schema,device, question, max_length=128) :
    input_text = f"Schema: {schema}, Question: {question}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql

   
def main():

    # Entry point: trigger dataset download and extraction.
    download_and_extract()

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

    # Fine-tuning T5 for SQL generation
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    args= TrainingArguments(
        output_dir= './sql_model',
        learning_rate= 3e-4,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        save_steps=50,
        disable_tqdm=True,
        logging_steps=50,  
        # ✅ Fix for XLA/TPU device — disable fused optimizer
        optim="adamw_torch",          # explicitly use standard AdamW
        no_cuda=False,                # let it use whatever accelerator is available
    )

    trainer= Trainer(
        model=model,
        args=args,
        train_dataset= tokenized_data,
        callbacks=[TimingCallback()]
    )

    trainer.train()
    trainer.save_model("./sql_model")
    tokenizer.save_pretrained("./sql_model")

    tokenizer = T5Tokenizer.from_pretrained("./sql_model")
    model = T5ForConditionalGeneration.from_pretrained("./sql_model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()

    userwish = input("Enter your question (or 'exit' to quit): ")

    while userwish.lower() != "exit":

        schema = input("Enter the table schema (e.g., 'name(str), age(int), salary(float)'): ")
        generated_sql = generate_sql(tokenizer, model, schema, device, userwish)
        print("\nGenerated SQL Query:\n", generated_sql)

        if is_valid_sql(generated_sql):
             print("Generated SQL:", generated_sql)
        else:
            print("The generated SQL query is invalid.")

        userwish = input("\nEnter another question (or 'exit' to quit): ")
    

    print("end of main")


if __name__ == "__main__":
    main()
