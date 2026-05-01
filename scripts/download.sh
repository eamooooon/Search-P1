export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120

save_path=./cache

python scripts/download.py --save_path $save_path

cat $save_path/part_* > $save_path/e5_Flat.index

gzip -d $save_path/wiki-18.jsonl.gz

modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./models/Qwen2.5-3B-Instruct
modelscope download --model intfloat/e5-base-v2 --local_dir ./models/e5-base-v2