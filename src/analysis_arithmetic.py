import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
from eval_llama import add_parsed_output_to_df
from llama2.parse import build_parser


def main():
	parser = build_parser('arithmetic')
	models_names = [('llama2-7b-chat', 'zero_shot'), ('llama2-13b-chat', 'zero_shot'), ('llama2-70b-chat', 'zero_shot'),
					# ('llama2-7b-chat', 'zero_shot_cot'), ('llama2-13b-chat', 'zero_shot_cot'), ('llama2-70b-chat', 'zero_shot_cot'),
					('mammoth-7b', 'zs_mammoth'), ('mammoth-13b', 'zs_mammoth'), ('mammoth-70b', 'zs_mammoth'),
					('metamath-7b', 'zs_metamath'), ('metamath-13b', 'zs_metamath'), ('metamath-70b', 'zs_metamath')]

	tables = {
		(model_name, prompt_type): load_table(model_name, prompt_type, 'arithmetic')
		for (model_name, prompt_type) in models_names
	}

	tables = reduce_tables(tables)
	tables = add_stats_to_tables(tables)
	plot_count_errors_by_num_digits(tables)
	plot_count_errors_by_op(tables)


def load_table(model_name, prompt_type, task_name):
	if os.path.exists(f'../out/models/{model_name}/{task_name}_{prompt_type}.csv'):
		df = pd.read_csv(f'../out/models/{model_name}/{task_name}_{prompt_type}.csv', index_col=0)
		return df
	else:
		print(f'File ../out/models/{model_name}/{task_name}_{prompt_type}.csv not found.')
		return None


def reduce_tables(tables):
	for k, df in tables.items():
		if df is not None:
			tables[k] = df[df['difficulty_split'] == 'N1_O2']
	return tables


def plot_count_errors_by_num_digits(tables):
	fig, axes = plt.subplots(len(tables)//3, 3, figsize=(7, 8), sharex=True, sharey=True)

	for ((model_name, prompt_type), table), ax in zip(tables.items(), axes.flat):
		if table is not None:
			ax = sns.histplot(data=table, x='count_dd', hue='is_exact', multiple='stack', ax=ax)
			ax.set_title(get_model_size(model_name))

		if '7b' in model_name:
			if 'llama' in model_name:
				ax.set_ylabel(get_model_family_name(model_name, prompt_type))
			else:
				ax.set_ylabel(get_model_family_name(model_name))

	plt.savefig('../out/plots/count_errors_by_num_digits_arithmetic.pdf', bbox_inches='tight')


def plot_count_errors_by_op(tables):
	fig, axes = plt.subplots(len(tables)//3, 3, figsize=(7, 8), sharex=True, sharey=True)

	for ((model_name, prompt_type), table), ax in zip(tables.items(), axes.flat):
		if table is not None:
			ax = sns.histplot(data=table, x='has_prod', hue='is_exact', multiple='stack', ax=ax)
			ax.set_title(get_model_size(model_name))

		if '7b' in model_name:
			if 'llama' in model_name:
				ax.set_ylabel(get_model_family_name(model_name, prompt_type))
			else:
				ax.set_ylabel(get_model_family_name(model_name))

	plt.savefig('../out/plots/plot_count_errors_by_op.pdf', bbox_inches='tight')


def add_stats_to_tables(tables):
	parser = build_parser('arithmetic')

	for k, df in tables.items():
		if df is not None:
			add_parsed_output_to_df(df, parser)
			df['is_exact'] = df['original_target'] == df['parsed_output']
			tables[k] = df

	for k, df in tables.items():
		if df is not None:
			double_digit_number_re = re.compile(r'\d\d')
			df['count_dd'] = df['original_input'].apply(double_digit_number_re.findall).apply(len)
			df['has_prod'] = df['original_input'].apply(lambda x: x.count('*'))
			tables[k] = df
	
	return tables


def get_model_size(model_name):
	if '7b' in model_name:
		return '7B'

	elif '13b' in model_name:
		return '13B'

	elif '70b' in model_name:
		return '70B'


def get_model_family_name(model_name, prompt_type=None):
	if 'llama' in model_name:
		if prompt_type is None:
			return 'Llama2'
		else:
			return 'Llama2\n' + prompt_type
	elif 'mammoth' in model_name:
		return 'MAmmoTH'
	elif 'metamath' in model_name:
		return 'Metamath'


if __name__ == '__main__':
	main()