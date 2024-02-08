HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path google/mt5-small \
    --max_source_length 1024 \
    --max_target_length 64 \
    --output_dir model/mt5-hin-small/output \
    --cache_dir model/mt5-hin-small/cache/ \
    --source_prefix "summarize: " \
    --do_train \
    --num_train_epochs 10 \
    --train_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/clean_hindi_train.csv \
    --use_lineterminator True\
    --per_device_train_batch_size 4 \
    --do_eval \
    --validation_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/clean_hindi_val.csv \
    --evaluation_strategy "epoch" \
    --per_device_eval_batch_size 4 \
    --num_beams 4 \
    --predict_with_generate \
    --dataloader_num_workers 1 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 42 \
    --use_fast_tokenizer False \
    --summary_column summary \
    --text_column article $@
    # --max_train_samples 4 \
    # --max_eval_samples 4 \
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --do_train \
    # --overwrite_output_dir \
