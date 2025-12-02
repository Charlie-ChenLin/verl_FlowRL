mkdir -p ./data/search_r1_retriever
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py \
  --save_path ./data/search_r1_retriever
cat ./data/search_r1_retriever/part_a* > ./data/search_r1_retriever/e5_Flat.index

cd /mnt/shared-storage-user/chenlin1/verl_FlowRL_lchen
export PYTHONPATH=$PWD:$PYTHONPATH
python examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir ./data/searchR1_processed_direct


# 下载 intfloat/e5-base-v2 到指定目录
huggingface-cli download intfloat/e5-base-v2 --local-dir ./models/e5-base-v2 --local-dir-use-symlinks False
