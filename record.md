参数	原来（8卡）	现在（4卡）
CUDA_VISIBLE_DEVICES	0-7	0-3
trainer.n_gpus_per_node	8	4
data.train_batch_size	512	256
data.val_batch_size	256	128
ppo_mini_batch_size	256	128
ppo_micro_batch_size	64	32
log_prob_micro_batch_size	128	64