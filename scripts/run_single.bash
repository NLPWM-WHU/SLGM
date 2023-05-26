source scripts/base_absa.bash
export CUDA_VISIBLE_DEVICES=0
export dataset=laptop14
export seed=42
export output_dir=./outputs/${dataset}_${seed}

bash scripts/base_run.bash

