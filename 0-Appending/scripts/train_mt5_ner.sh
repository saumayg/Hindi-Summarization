# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path google/mt5-small \
    --max_source_length 1024 \
    --max_target_length 64 \
    --output_dir model/mt5-hin-small-ner/output \
    --cache_dir model/mt5-hin-small-ner/cache/ \
    --source_prefix ". summarize: " \
    --ner_prefix "entity: " \
    --do_train \
    --num_train_epochs 1 \
    --train_file ../../Datasets/Hindi_summarization/XLSum/NER/hindi_train_ner_1024.csv \
    --max_train_samples 4 \
    --per_device_train_batch_size 1 \
    --do_eval \
    --validation_file ../../Datasets/Hindi_summarization/XLSum/NER/hindi_val_ner_1024.csv \
    --evaluation_strategy "epoch" \
    --max_eval_samples 4 \
    --per_device_eval_batch_size 1 \
    --num_beams 4 \
    --predict_with_generate \
    --dataloader_num_workers 1 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 42 \
    --use_fast_tokenizer False \
    --do_ner \
    --ner_column ner_str \
    --summary_column summary \
    --text_column text $@
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --do_train \
    # --overwrite_output_dir \
