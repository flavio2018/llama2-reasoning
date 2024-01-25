import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def main():

	models_names = [('llama2-7b-chat', 'zero_shot'), ('llama2-13b-chat', 'zero_shot'), ('llama2-70b-chat', 'zero_shot'),
					# ('llama2-7b-chat', 'zero_shot_cot'), ('llama2-13b-chat', 'zero_shot_cot'), ('llama2-70b-chat', 'zero_shot_cot'),
					('mammoth-7b', 'zs_mammoth'), ('mammoth-13b', 'zs_mammoth'), ('mammoth-70b', 'zs_mammoth'),
					('metamath-7b', 'zs_metamath'), ('metamath-13b', 'zs_metamath'), ('metamath-70b', 'zs_metamath')]

	tables_by_task = {
		'listops': {
			(model_name, prompt_type): load_table(model_name, prompt_type, 'listops')
			for (model_name, prompt_type) in models_names
		},
		'arithmetic': {
			(model_name, prompt_type): load_table(model_name, prompt_type, 'arithmetic')
			for (model_name, prompt_type) in models_names
		},
		'algebra': {
			(model_name, prompt_type): load_table(model_name, prompt_type, 'algebra')
			for (model_name, prompt_type) in models_names
		},
	}

	plot_tables_listops(tables_by_task['listops'])
	plot_tables_arit_alg(tables_by_task['arithmetic'], 'arithmetic')
	# plot_tables_arit_alg(tables_by_task['algebra'], 'algebra')


def load_table(model_name, prompt_type, task_name):
	if os.path.exists(f'../out/accuracy_tables/{model_name}/{task_name}_{prompt_type}.csv'):
		df = pd.read_csv(f'../out/accuracy_tables/{model_name}/{task_name}_{prompt_type}.csv', index_col=0)
		return reformat_floats(df)
	else:
		print(f'File ../out/accuracy_tables/{model_name}/{task_name}_{prompt_type}.csv not found.')
		if task_name == 'listops':
			return pd.DataFrame([[np.nan]*3]*4)
		else:
			return pd.DataFrame([np.nan]*4)


def reformat_floats(df):
    return df.astype(str).map(lambda x: x.replace(',', '.')).astype(float)


def plot_tables_listops(tables):
	fig, axes = plt.subplots(len(tables)//3, 3, figsize=(5, 5), sharex=True, sharey=True)

	for ((model_name, prompt_type), table), ax in zip(tables.items(), axes.flat):
		ax = sns.heatmap(table.iloc[:, ::-1].T, ax=ax, square=True, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': 6}, cbar=False, )
		ax.set_title(get_model_size(model_name))

		if '7b' in model_name:
			if 'llama' in model_name:
				ax.set_ylabel(get_model_family_name(model_name, prompt_type))
			else:
				ax.set_ylabel(get_model_family_name(model_name))

	# plt.suptitle('listops')
	plt.savefig('../out/plots/accuracy_tables_listops.pdf', bbox_inches='tight')


def plot_tables_arit_alg(tables, task_name):
	fig, axes = plt.subplots(len(tables)//3, 3, figsize=(5, 3), sharex=True, sharey=True)

	for ((model_name, prompt_type), table), ax in zip(tables.items(), axes.flat):
		ax = sns.heatmap(table.T, ax=ax, square=True, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': 6}, cbar=False, )
		ax.set_title(get_model_size(model_name))

		if '7b' in model_name:
			if 'llama' in model_name:
				ax.set_ylabel(get_model_family_name(model_name, prompt_type))
			else:
				ax.set_ylabel(get_model_family_name(model_name))
	
	# plt.suptitle(task_name)
	plt.savefig(f'../out/plots/accuracy_tables_{task_name}.pdf', bbox_inches='tight')


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
