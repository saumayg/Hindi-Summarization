python rouge_scoring.py \
    --prediction_file ../2-Multi-training/results/5/generated_predictions_val.csv \
    --prediction_column Predictions \
    --summary_file ../Datasets/Hindi_summarization/XLSum/clean_hindi_val.csv \
    --summary_column text

python bert_scoring.py \
    --prediction_file ../2-Multi-training/results/5/generated_predictions_val.csv \
    --prediction_column Predictions \
    --summary_file ../Datasets/Hindi_summarization/XLSum/clean_hindi_val.csv \
    --summary_column text