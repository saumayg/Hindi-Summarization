import evaluate
import argparse
import pandas as pd

from dataclasses import field
from typing import Optional

def read_file(filename, column):
    ext = filename.split(".")[-1]
    text = []
    if ext == "txt":
        with open(filename) as f:
            text = [line.rstrip() for line in f]
    else:
        df = pd.read_csv(filename)
        # If no column name passed, considering first column as the summary column
        if column is None:
            text = df.iloc[:, 0].tolist()
        else:
            text = df[column].tolist()
    
    return text
        

parser = argparse.ArgumentParser()

parser.add_argument("--prediction_file", type=str, required=True, help="File containing the predicted summaries by the model")
parser.add_argument("--summary_file", type=str, required=True, help="File containing the golden summaries")
parser.add_argument("--prediction_column", type=str, required=False, default=None, help="The name of the column in the datasets containing the predicted summary")
parser.add_argument("--summary_column", type=str, required=False, default=None, help="The name of the column in the datasets containing the golden summary")
parser.add_argument("--stemmer", type=bool, required=False, default=True, help="Boolean telling to use stemmer for the rouge score calculation")
parser.add_argument("--lang", type=str, required=False, default="hindi", help="Language to use for multilingual rouge score calculation")

args = parser.parse_args()

metric = evaluate.load('/home/g100may/MTP/NEA-ATS/Evaluation/rouge.py')

pred_ext = args.prediction_file.split(".")[-1]
assert pred_ext in ["txt", "csv"], "'prediction file' should be a txt or csv file"
summ_ext = args.summary_file.split(".")[-1]
assert summ_ext in ["txt", "csv"], "'summary file' should be a txt or csv file"

pred_txt = read_file(args.prediction_file, args.prediction_column)
summ_txt = read_file(args.summary_file, args.summary_column)

results = metric.compute(predictions=pred_txt, references=summ_txt, use_stemmer=args.stemmer, language=args.lang)
results = {k: round(v*100, 4) for k, v in results.items()}

print(results)