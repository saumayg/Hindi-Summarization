python ./run_summarization.py \
    --model_name_or_path ./mt5-base/output/checkpoint-62910 \
    --max_source_length 1024 \
    --max_target_length 64 \
    --output_dir ./mt5-base/output \
    --source_prefix "summarize: " \
    --do_predict \
    --test_file hindi_val.csv \
    --num_beams 4 \
    --predict_with_generate \
    --dataloader_num_workers 1 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 42 \
    --use_fast_tokenizer False \
    --summary_column summary \
    --resume_from_checkpoint ./mt5-base/output/checkpoint-62910 \
    --text_column text $@
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --do_train \
    # --overwrite_output_dir \
