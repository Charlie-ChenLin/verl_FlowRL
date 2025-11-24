mkdir -p ./data/search_r1_retriever
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py \
  --save_path ./data/search_r1_retriever
cat ./data/search_r1_retriever/part_a* > ./data/search_r1_retriever/e5_Flat.index
