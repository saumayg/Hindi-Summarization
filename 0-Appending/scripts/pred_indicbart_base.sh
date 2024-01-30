python ./run_summarization.py \
    --model_name_or_path ./indicbart-base/output/checkpoint-125810 \
    --lang hi_IN \
    --forced_bos_token=tokenizer.lang_code_to_id["hi_IN"] \
    --max_source_length 1024 \
    --max_target_length 40 \
    --output_dir ./indicbart-base/output \
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
    --resume_from_checkpoint ./indicbart-base/output/checkpoint-125810 \
    --text_column text $@
    # --do_train \
    # --overwrite_output_dir \
