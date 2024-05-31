HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path google/mt5-base \
    --max_source_length 1024 \
    --max_target_length 84 \
    --output_dir model/lsn/mt5-base/output \
    --cache_dir model/cache/mt5-base \
    --lr_scheduler_type="linear" \ done
    --learning_rate=5e-4 \ done
    --warmup_steps 250 \ done
    --weight_decay 0.01 \ done
    --gradient_accumulation_steps=16 \ done
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
    --save_strategy "epoch" \
    --logging_first_step \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --seed 42 \
    --summary_column summary \
    --text_column article $@
    # --max_train_samples 4 \
    # --max_eval_samples 4 \
    # --logging_strategy "epoch" \
    # --source_prefix "summarize: " \
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --do_train \
    # --overwrite_output_dir \
