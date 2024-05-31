HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path google/mt5-base \
    --max_source_length 1024 \
    --max_target_length 84 \
    --output_dir model/lsn/mt5-base-ner/output \
    --cache_dir model/cache/mt5-base \
    --lr_scheduler_type="linear" \
    --learning_rate=5e-4 \
    --warmup_steps 250 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps=16 \
    --source_prefix ".summarize: " \
    --ner_prefix "entity: " \
    --do_train \
    --num_train_epochs 10 \
    --train_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/NER/hindi_train_ner_1024.csv \
    --use_lineterminator True \
    --per_device_train_batch_size 2 \
    --do_eval \
    --validation_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/NER/hindi_val_ner_1024.csv \
    --evaluation_strategy "epoch" \
    --per_device_eval_batch_size 1 \
    --num_beams 4 \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --predict_with_generate \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 42 \
    --do_ner \
    --ner_column ner_str \
    --summary_column summary \
    --resume_from_checkpoint model/lsn/mt5-base-ner/output/checkpoint-8168 \
    --text_column article $@
    # --max_train_samples 4 \
    # --max_eval_samples 4 \
    # --do_train \
    # --overwrite_output_dir \
