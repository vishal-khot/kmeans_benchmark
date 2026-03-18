# kmeans_benchmark

bench_kmeans.py
---------------
Benchmark for K-Means clustering kernels used in sparse attention
(DSA lightning indexer / SparseVideoGen2 pipeline).
 
Dimensions sourced from DeepSeek-V3.2 config:
 * num_attention_heads : 128
 * index_n_heads       : 64
 * index_head_dim      : 128
 * kv_lora_rank        : 512
 * FP8 fmt             : E4M3
 
Measures:
  1. Correctness  -- Triton BF16 vs PyTorch BF16
                  -- Triton FP8  vs PyTorch BF16 (compare precision loss)
  2. Latency      -- triton.testing.do_bench 
  3. Memory BW    -- ncu
 
Usage:
  # Quick run (correctness + latency):
  python bench_kmeans.py
 
  # ncu profiling:
  ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
      --kernel-name-base kmeans \
      --target-processes all \
      python bench_kmeans.py --profile
