# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_summarization.py \
    --model_name_or_path model/ilsum2023/mb50-base/output/checkpoint-2650 \
    --lang hi_IN \
    --forced_bos_token hi_IN \
    --max_source_length 1024 \
    --max_target_length 90 \
    --output_dir model/ilsum2023/mb50-base/output \
    --do_predict \
    --test_file ../../Datasets/Hindi_summarization/ILSUM2023/clean_hindi_val.csv \
    --predict_with_generate \
    --save_strategy "epoch" \
    --logging_strategy "epoch" \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --seed 42 \
    --summary_column Summary \
    --text_column Article $@
    # --max_train_samples 4 \
    # --max_eval_samples 4 \
    # --logging_strategy "epoch" \
    # --source_prefix "summarize: " \
    # --use_lineterminator True\
    # --resume_from_checkpoint model/indicbartss-hin/ \
    # --do_train \
    # --overwrite_output_dir \
