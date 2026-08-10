# Examples Overview

This directory contains runnable templates for data preprocessing, SFT, RL agent
training, retrieval services, multi-node launch, Ray/Slurm setup, and placement
experiments. Most training scripts are shell wrappers around `verl.trainer`
entrypoints with concrete model, data, environment, and algorithm settings.

## Recommended starting points

For a quick training-chain smoke test, start with one of:

- `grpo_trainer/run_webshop.sh`: text-only WebShop agent training with GRPO.
- `gigpo_trainer/run_webshop.sh`: text-only WebShop agent training with GIGPO.
- `grpo_trainer/run_sokoban.sh`: visual Sokoban agent training with a VL model.

The PPO examples are also useful, but they include a critic and usually need
more memory and tuning.

## PPO trainer

- `ppo_trainer/run_alfworld.sh`: standard PPO on the ALFWorld text interaction
  environment. It uses GAE and trains a separate critic.
- `ppo_trainer/run_webshop.sh`: standard PPO on the WebShop text shopping
  environment. It also uses GAE and a separate critic.

## GRPO trainer

- `grpo_trainer/run_alfworld.sh`: GRPO on ALFWorld. It uses grouped rollouts for
  relative advantage estimation instead of a separately trained critic.
- `grpo_trainer/run_webshop.sh`: GRPO on WebShop.
- `grpo_trainer/run_sokoban.sh`: GRPO on the visual Sokoban environment with
  `Qwen/Qwen2.5-VL-3B-Instruct`.
- `grpo_trainer/run_balckjack.sh`: GRPO on the visual Blackjack card
  environment. The filename contains a typo, but the target environment is
  Blackjack.

## GIGPO trainer

- `gigpo_trainer/run_alfworld.sh`: GIGPO on ALFWorld with
  `Qwen/Qwen2.5-1.5B-Instruct`.
- `gigpo_trainer/run_webshop.sh`: GIGPO on WebShop with
  `Qwen/Qwen2.5-1.5B-Instruct`.
- `gigpo_trainer/run_sokoban.sh`: GIGPO on visual Sokoban with
  `Qwen/Qwen2.5-VL-3B-Instruct`.
- `gigpo_trainer/run_blackjack.sh`: GIGPO on the visual Blackjack card
  environment.
- `gigpo_trainer/run_ezpoints.sh`: GIGPO on the visual EZPoints card/number
  reasoning environment.
- `gigpo_trainer/run_numberline.sh`: GIGPO on the visual NumberLine environment
  with `Qwen/Qwen2-VL-2B-Instruct`.
- `gigpo_trainer/run_search.sh`: GIGPO for search-augmented reasoning. It
  expects a retriever service at `http://127.0.0.1:8000/retrieve`.
- `gigpo_trainer/run_alfworld_lora.sh`: LoRA GIGPO on ALFWorld with
  `Qwen/Qwen2.5-7B-Instruct`.
- `gigpo_trainer/run_webshop_lora.sh`: LoRA GIGPO on WebShop with
  `Qwen/Qwen2.5-7B-Instruct`.
- `gigpo_trainer/run_webshop_qwen3.sh`: GIGPO on WebShop with `Qwen/Qwen3-1.7B`.
- `gigpo_trainer/run_sokoban_qwen3vl.sh`: GIGPO on visual Sokoban with
  `Qwen/Qwen3-VL-2B-Instruct`.

## Dynamic GIGPO trainer

These examples use `algorithm.adv_estimator=gigpo` with dynamic advantage
normalization modes such as `mean_norm` or `mean_std_norm`.

- `gigpo_dynamic_trainer/run_alfworld.sh`: dynamic GIGPO on ALFWorld.
- `gigpo_dynamic_trainer/run_webshop.sh`: dynamic GIGPO on WebShop.
- `gigpo_dynamic_trainer/run_sokoban.sh`: dynamic GIGPO on visual Sokoban.

## DAPO trainer

These scripts use the PPO entrypoint with GRPO-style advantage estimation plus
DAPO-style settings such as custom clipping ranges and group filtering.

- `dapo_trainer/run_alfworld.sh`: DAPO-style training on ALFWorld.
- `dapo_trainer/run_webshop.sh`: DAPO-style training on WebShop.

## RLOO trainer

- `rloo_trainer/run_alfworld.sh`: RLOO on ALFWorld. RLOO uses a leave-one-out
  baseline over grouped samples.
- `rloo_trainer/run_webshop.sh`: RLOO on WebShop.

## GSPO trainer

- `gspo_trainer/run_alfworld.sh`: GSPO-style ALFWorld experiment. The script
  still routes through the PPO entrypoint and uses `algorithm.adv_estimator=grpo`,
  while the experiment name marks the GSPO variant.

## SFT

- `sft/gsm8k/run_qwen_05_sp2.sh`: SFT for `Qwen/Qwen2.5-0.5B-Instruct` on
  GSM8K with Ulysses sequence parallel size 2.
- `sft/gsm8k/run_qwen_05_sp2_liger.sh`: the same Qwen 0.5B GSM8K SFT template
  with `liger-kernel` enabled.
- `sft/gsm8k/run_qwen_05_peft.sh`: LoRA/PEFT SFT for Qwen 0.5B on GSM8K.
- `sft/gsm8k/run_gemma_2b.sh`: SFT for `google/gemma-2b-it` on GSM8K.
- `sft/gsm8k/run_gemma_7b.sh`: SFT for `google/gemma-1.1-7b-it` on GSM8K.
- `sft/gsm8k/run_deepseek_6b7.sh`: SFT for
  `deepseek-ai/deepseek-coder-6.7b-instruct` on GSM8K.
- `sft/multiturn/run_qwen_05_sp2.sh`: SFT for multi-turn chat data with
  `data.multiturn.enable=true`.

## Data preprocessing

- `data_preprocess/prepare.py`: lightweight text/visual parquet generator used
  by many agent training examples. It uses Geometry3K only as a convenient source
  for modality and sample count.
- `data_preprocess/geo3k.py`: converts Geometry3K to parquet.
- `data_preprocess/gsm8k.py`: converts GSM8K to parquet for SFT/RL workflows.
- `data_preprocess/gsm8k_multiturn_w_tool.py`: converts GSM8K to a multi-turn
  tool-calling format with `calc_gsm8k_reward` tool metadata.
- `data_preprocess/math_dataset.py`: converts the MATH/lighteval dataset to
  parquet.
- `data_preprocess/hellaswag.py`: converts HellaSwag multiple-choice data to
  parquet.
- `data_preprocess/multiturn.py`: creates a small synthetic multi-turn dataset
  for testing the multi-turn SFT path.
- `data_preprocess/full_hh_rlhf.py`: converts `Dahoas/full-hh-rlhf` into SFT,
  reward-model, or RL-style parquet data.
- `data_preprocess/dapo_multiturn_w_tool.py`: converts DAPO-Math-17k to a
  multi-turn tool-calling format.
- `data_preprocess/aime2024_multiturn_w_tool.py`: converts AIME2024 to a
  multi-turn tool-calling format.
- `data_preprocess/preprocess_search_r1_dataset.py`: converts Search-R1 style
  search datasets into parquet with `search` tool metadata.

## Generation

- `generation/run_deepseek_v2_lite_math.sh`: batch generation for math prompts
  with `verl.trainer.main_generation` on a single-node setup.
- `generation/run_deepseek7b_mutli_node.sh`: multi-node batch generation
  template. The filename contains a typo, but the script is for multi-node
  generation.

## Search

- `search/searchr1_download.py`: downloads the wiki index and corpus files used
  by Search-R1 style retrieval.
- `search/retriever/retrieval_server.py`: starts a retrieval HTTP service with
  BM25 or dense retrieval support.
- `search/retriever/retrieval_launch.sh`: launches the retriever with an E5
  model, FAISS GPU, and port `8000`.

## Environment server

- `env_server/start_appworld_server.sh`: starts many AppWorld environment server
  instances and records their ports in `appworld_ports.ports`. Use this for
  experiments that need external AppWorld environment services.

## Ray and Slurm

- `ray/tutorial.ipynb`: tutorial material for understanding Ray-based worker
  scheduling and remote execution.
- `slurm/ray_on_slurm.slurm`: Slurm template for starting a Ray cluster in a
  multi-node job.

## Prompt agent

- `prompt_agent/gpt4o_alfworld.py`: prompt-only ALFWorld baseline that calls the
  OpenAI API instead of training a local policy.
- `prompt_agent/run_gpt4o_agent.sh`: shell wrapper for the GPT-4o ALFWorld
  baseline.

## Split placement

The split placement example demonstrates custom Ray resource pools: actor and
reference policy can be placed on one GPU pool, while critic and reward model can
be placed on another. This is useful for experimenting with asynchronous PPO
execution and resource isolation.

- `split_placement/README.md`: detailed walkthrough for the split placement
  experiment.
- `split_placement/main_ppo_split.py`: PPO entrypoint modified for explicit
  resource-pool placement.
- `split_placement/split_monkey_patch.py`: monkey patch for the PPO training
  loop to demonstrate asynchronous operations.
- `split_placement/run_deepseek7b_llm.sh`: DeepSeek 7B split-placement run
  script.
- `split_placement/config/ppo_trainer_split.yaml`: configuration file for the
  split-placement PPO example.
