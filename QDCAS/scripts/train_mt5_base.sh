# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path agemagician/mlong-t5-tglobal-base \
    --max_source_length 4096 \
    --max_target_length 100 \
    --output_dir model/ilsum2023/mlongt5-base/output \
    --cache_dir model/cache/mlongt5-base \
    --lr_scheduler_type="linear" \
    --learning_rate=5e-4 \
    --warmup_steps 250 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps=16 \
    --do_train \
    --num_train_epochs 10 \
    --train_file ../../Datasets/Hindi_summarization/ILSUM2023/clean_hindi_train.csv \
    --train_crawl_file ../../Datasets/Hindi_summarization/ILSUM2023/ILSUM2023-train-crawl.jsonl \
    --per_device_train_batch_size 2 \
    --do_eval \
    --validation_file ../../Datasets/Hindi_summarization/ILSUM2023/clean_hindi_val.csv \
    --val_crawl_file ../../Datasets/Hindi_summarization/ILSUM2023/ILSUM2023-val-crawl.jsonl \
    --evaluation_strategy "epoch" \
    --logging_strategy "epoch" \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --save_strategy "epoch" \
    --logging_first_step \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --seed 42 \
    --summary_column Summary \
    --id_column Id \
    --text_column Article $@
    # --source_prefix "summarize: " \
    # --max_train_samples 400 \
    # --max_eval_samples 400 \
    # --use_lineterminator True\
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --overwrite_output_dir \
