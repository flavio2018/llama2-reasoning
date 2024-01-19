#!/usr/bin/env python


from gpt.data import build_input_target
from llama2 import HFInterface
from gpt.prompts import get_prompt_builder
import datetime
import hydra
import pandas as pd


@hydra.main(config_path="../conf/", config_name="query_llama", version_base='1.2')
def main(cfg):
	print(f"##### {cfg.task_name} nesting {cfg.nesting}, {cfg.n_operands} operands, {cfg.prompt_type} prompt, role {cfg.role}. #####")

	run_timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
	batch, target = build_input_target(
		**{
			"task_name": cfg.task_name,
			"batch_from": cfg.batch_from,
			"bs": cfg.bs,
			"nesting": cfg.nesting,
			"n_ops": cfg.n_operands
		}
	)

	role_description = get_role_description(cfg)

	prompt_builder = get_prompt_builder(cfg.task_name)
	prompts = prompt_builder.build_prompt(batch, cfg.prompt_type)
	hf_interface = HFInterface()

	if (cfg.prompt_type == 'zero_shot_cot') or (cfg.prompt_type == 'self_consistency'):
		outputs = hf_interface.query_model_zero_shot_cot(prompts)
	else:
		outputs = hf_interface.query_model(prompts, role_description)

	if cfg.role:
		cfg.prompt_type = cfg.prompt_type + '_role'

	print("Dumping run DataFrame...")
	df = pd.DataFrame(
		columns=["task_name",
				 "prompt_type",
				 "prompt",
				 "gpt_output",
				 "original_input",
				 "original_target"]
	)

	df['prompt'] = prompts 
	df['original_input'] = batch
	df['original_target'] = target
	df['task_name'] = cfg.task_name
	df['prompt_type'] = cfg.prompt_type
	df['gpt_output'] = outputs

	df.to_csv(f'../out/llama2/{cfg.task_name}__nes{cfg.nesting}__nop{cfg.n_operands}__{cfg.prompt_type}__{cfg.batch_from}+{cfg.bs}__{run_timestamp}.csv')
	print("Done.")


def get_role_description(cfg):
	if cfg.role == True:
		return "You are a brilliant mathematician."
	else:
		return None


if __name__ == '__main__':
	main()
