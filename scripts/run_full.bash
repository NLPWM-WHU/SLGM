source scripts/base_absa.bash
export CUDA_VISIBLE_DEVICES=0
for dataset in laptop14 rest14 rest15 rest16;do
  for seed in 42 43 44 45 46;do
      export dataset=${dataset}
      export seed=${seed}
      export output_dir=./outputs/${dataset}_${seed}

      bash scripts/base_run.bash
  done
done
