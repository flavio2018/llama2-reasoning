from llama2.data import load_test_df
from llama2 import HFInterface
from llama2.prompts import get_prompt_builder
import pandas as pd
import os
import tqdm


class ModelQuerierOnTask:

	def __init__(self, cfg):
		self.model_name = cfg.model_name
		self.task_name = cfg.task_name
		self.prompt_type = cfg.prompt_type
		self.difficulty_split = cfg.difficulty_split
		self.hfi = HFInterface()
		self.prompt_builder = get_prompt_builder(self.task_name)
		self.test_dataset_df = load_test_df(self.task_name)
		self.slice_difficulty_split()
		self.outputs_df = pd.DataFrame(
			columns=["task_name",
					 "prompt_type",
					 "prompt",
	             	 "0_shot_cot_first_out",
					 "model_output",
					 "original_input",
					 "original_target",
					 "difficulty_split"]
		)
		print(f"Created querier for {self.model_name} on {self.task_name} {self.difficulty_split} with {self.prompt_type} prompting.")

	def query_model(self):
		if not self.load_outputs_df():
			return

		for sample_idx in tqdm.trange(len(self.outputs_df), len(self.test_dataset_df)):
			sample = self.test_dataset_df.iloc[sample_idx]['X']
			target = self.test_dataset_df.iloc[sample_idx]['Y']
			nesting = self.test_dataset_df.iloc[sample_idx]['nesting']
			num_operands = self.test_dataset_df.iloc[sample_idx]['num_operands']
			difficulty_split = f"N{nesting}_O{num_operands}"
			prompt = self.prompt_builder.build_prompt([sample], self.prompt_type)[0]

			if (self.prompt_type == 'zero_shot_cot'):
				output = self.hfi.query_model_zero_shot_cot([prompt])[0]
				zero_shot_cot_first_out = self.hfi.zero_cot_first_outputs[0]
			elif ('mammoth' in self.model_name) or ('metamath' in self.model_name):
				output = self.hfi.query_model([prompt], None, structured_prompt=False)[0]
				zero_shot_cot_first_out = None
			else:
				output = self.hfi.query_model([prompt], None)[0]
				zero_shot_cot_first_out = None
			
			curr_outputs_df = pd.DataFrame([[self.task_name, self.prompt_type, prompt, zero_shot_cot_first_out, output, sample, target, difficulty_split]], columns=self.outputs_df.columns)
			self.outputs_df = pd.concat([self.outputs_df, curr_outputs_df], ignore_index=True)

			if sample_idx % 10 == 0:
				self.dump_outputs_df()

		self.dump_outputs_df()

	def slice_difficulty_split(self):
		if self.difficulty_split == '':
			return
		nesting, num_operands = self.difficulty_split.split('_')
		nesting, num_operands = int(nesting[1]), int(num_operands[1])
		self.test_dataset_df = self.test_dataset_df[(self.test_dataset_df['nesting'] == nesting) & (self.test_dataset_df['num_operands'] == num_operands)]


	def load_outputs_df(self):
		if os.path.exists(f"{self.base_dir}/{self.task_name}_{self.prompt_type}{self.difficulty_split}.csv"):
			print("Found existing output df.")
			outputs_df = pd.read_csv(f"{self.base_dir}/{self.task_name}_{self.prompt_type}{self.difficulty_split}.csv", index_col=0)

			if len(outputs_df) < len(self.test_dataset_df):
				print("Found less samples than in test set. Loading df and resuming run.")
				self.outputs_df = outputs_df
				return True
			else:
				print("Found same number of samples as in test set. Aborting run.")
				return False

		print("No previous run outputs found. Starting run from scratch.")
		return True

	def dump_outputs_df(self):
		print(f"Dumping outputs DataFrame with {len(self.outputs_df)} samples...")
		self.outputs_df.to_csv(f'{self.base_dir}/{self.task_name}_{self.prompt_type}{self.difficulty_split}.csv')
		print("Done.")

	@property
	def base_dir(self):
		return f"../out/models/{self.model_name}"
