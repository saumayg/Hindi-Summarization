HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python ../run_mlm_seq2seq.py \
    --model_name_or_path ai4bharat/IndicBART \
    --lang hi_IN \
    --forced_bos_token=tokenizer.lang_code_to_id["hi_IN"] \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --output_dir model/lsn/indicbart-mlm/5/output \
    --cache_dir model/cache/indicbart-mlm \
    --do_train \
    --num_train_epochs 10 \
    --train_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/clean_hindi_train.csv \
    --ner_train_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/NER/ner_hindi_train_1024.json \
    --use_lineterminator True\
    --per_device_train_batch_size 2 \
    --do_eval \
    --validation_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/clean_hindi_val.csv \
    --ner_val_file ../../Datasets/Hindi_summarization/Long-short-news-dataset/NER/ner_hindi_val_1024.json \
    --evaluation_strategy "epoch" \
    --per_device_eval_batch_size 1 \
    --num_beams 4 \
    --predict_with_generate \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --seed 42 \
    --resume_from_checkpoint model/lsn/indicbart-mlm/5/output/checkpoint-58084 \
    --text_column article $@
    # --max_train_samples 40 \
    # --max_eval_samples 40 \
    # --do_train \
    # --overwrite_output_dir \

