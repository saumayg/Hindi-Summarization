# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python extractive_gae.py \
    --pt_name ai4bharat/IndicBART \
    --train_dataset ../Datasets/Hindi_summarization/XLSum/clean_hindi_train.csv  \
    --article text \
    --summary summary \
    --emb_dim 1024 \
    --doc_threshold 0.950 \
    --sent_threshold 0.95 $@