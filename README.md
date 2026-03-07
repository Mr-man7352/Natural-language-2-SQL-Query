# NL → SQL

Fine-tune a T5-small model on the [WikiSQL](https://github.com/salesforce/WikiSQL) dataset to translate natural language questions into SQL queries.

---

## Project Structure

```
nl2sql/
├── src/
│   ├── download.py       # Dataset download & extraction
│   ├── preprocess.py     # SQL conversion & training-pair builder
│   ├── dataset.py        # PyTorch Dataset wrapper
│   ├── train.py          # Fine-tuning logic & timing callback
│   └── inference.py      # SQL generation & validation
├── config.py             # All hyperparameters and paths
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model
Downloads the WikiSQL dataset and fine-tunes T5-small (~8 000 examples, 2 epochs):

```bash
python main.py --train
```

### Run inference
Starts an interactive loop where you provide a schema and a question:

```bash
python main.py --infer
```

### Train then infer in one go

```bash
python main.py --train --infer
```

---

## Example

```
Table schema: name(text), age(real), salary(real)
Question: What is the average salary of employees older than 30?

Generated SQL:
  SELECT AVG(salary) FROM table WHERE age>'30'
```

---

## Configuration

All hyperparameters live in `config.py` — no need to dig into source files to change batch size, learning rate, etc.

| Parameter | Default |
|---|---|
| Model | t5-small |
| Epochs | 2 |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Max train samples | 8 000 |
