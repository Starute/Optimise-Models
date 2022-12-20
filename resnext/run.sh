#!/bin/bash

# nohup ./run.sh > log.txt &

# nsys profile -o profile/resnextBaselineProfile -t cuda,nvtx --force-overwrite true python3 resnext.py 

nsys profile -o profile/resnextHalfProfile -t cuda,nvtx --force-overwrite true python3 resnext.py --half_precision 

nsys profile -o profile/resnext0WorkerProfile -t cuda,nvtx --force-overwrite true python3 resnext.py --num_worker 0

nsys profile -o profile/resnext1WorkerProfile -t cuda,nvtx --force-overwrite true python3 resnext.py --num_worker 1

nsys profile -o profile/resnext4WorkerProfile -t cuda,nvtx --force-overwrite true python3 resnext.py --num_worker 4

nsys profile -o profile/resnext8WorkerProfile -t cuda,nvtx --force-overwrite true python3 resnext.py --num_worker 8

nsys profile -o profile/resnextBaselineProfile --cuda-memory-usage true -t cuda,nvtx --force-overwrite true python3 resnext.py 

nsys profile -o profile/resnextCheckpointProfile --cuda-memory-usage true -t cuda,nvtx --force-overwrite true python3 resnext_checkpoint.py 

nsys profile -o profile/resnextBaselineLargeBatchProfile --cuda-memory-usage true -t cuda,nvtx --force-overwrite true python3 resnext.py --batch_size 512

nsys profile -o profile/resnextCheckpointLargeBatchProfile --cuda-memory-usage true -t cuda,nvtx --force-overwrite true python3 resnext_checkpoint.py --batch_size 896

