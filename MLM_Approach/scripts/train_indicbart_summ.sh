HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path model/lsn/indicbart-mlm/5/output/checkpoint-145210 \
    --lang hi_IN \
    --forced_bos_token=tokenizer.lang_code_to_id["hi_IN"] \
    --max_source_length 1024 \
    --max_target_length 64 \
    --output_dir model/lsn/indicbart-summ/output \
    --cache_dir model/cache/indicbart-summ \
    --lr_scheduler_type="linear" \
    --learning_rate=5e-4 \
    --warmup_steps 250 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps=16 \
    --do_train \
    --num_train_epochs 10 \
    --train_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/clean_hindi_train.csv \
    --use_lineterminator True \
    --per_device_train_batch_size 2 \
    --do_eval \
    --validation_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/clean_hindi_val.csv \
    --evaluation_strategy "epoch" \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 42 \
    --summary_column summary \
    --text_column article $@
    # --max_train_samples 40 \
    # --max_eval_samples 40 \
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --do_train \
    # --overwrite_output_dir \

