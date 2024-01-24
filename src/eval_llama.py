import hydra
import functools
import pandas as pd
from llama2.parse import build_parser
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application


@hydra.main(config_path="../conf/", config_name="eval_llama", version_base='1.2')
def main(cfg):
	model_name = cfg.model_name
	task_name = cfg.task_name
	prompt_type = cfg.prompt_type

	nesting_ranges = {
		'arithmetic': range(1, 5),
		'algebra': range(1, 5),
		'listops': range(1, 5),
	}

	num_operands_ranges = {
		'arithmetic': range(2, 3),
		'algebra': range(2, 3),
		'listops': range(2, 5)
	}

	if task_name == 'listops':
		accuracy_table = pd.DataFrame(index=[1, 2, 3, 4], columns=[2, 3, 4])
	else:
		accuracy_table = pd.DataFrame(index=[1, 2, 3, 4], columns=[2])

	model_outputs_filepath = f'../out/models/{model_name}/{task_name}_{prompt_type}.csv'
	print(f"Reading file {model_outputs_filepath}")
	df = pd.read_csv(model_outputs_filepath, index_col=0)
	parser = build_parser(task_name)
	
	for nesting in nesting_ranges[task_name]:
		for num_operands in num_operands_ranges[task_name]:
			difficulty_split = f'N{nesting}_O{num_operands}'
			difficulty_split_idx = df[df['difficulty_split'] == difficulty_split].index
			difficulty_slice = df.iloc[difficulty_split_idx, :].copy()

			print(f"Evaluating {prompt_type} for {task_name}, ({nesting}, {num_operands}).")
			add_parsed_output_to_df(difficulty_slice, parser)
			run_acc = eval_df(difficulty_slice)
			accuracy_table.loc[nesting, num_operands] = run_acc

	accuracy_table.to_csv(f'../out/accuracy_tables/{model_name}/{task_name}_{prompt_type}.csv')


def add_parsed_output_to_df(df, parser):
	if (parser.task_name == 'arithmetic') or (parser.task_name == 'listops'):
		try:
			df['model_output'] = df['model_output'].apply(str).apply(lambda s: s.replace('.0', '').replace('.', ''))
		except ValueError as err:
			print(err)
	else:
		df['model_output'] = df['model_output'].astype(str)

	df['parsed_output'] = df['model_output'].apply(parser.parse_outputs)
	
	if parser.task_name == 'algebra':
		df['parsed_output'] = df['parsed_output'].apply(expr_to_sympy_w_except)


def eval_df(df):
	if df['task_name'].iloc[0] == 'algebra':
		return eval_sym_df(df)
	else:
		return eval_str_df(df)


def eval_str_df(run_df):
	assert 'parsed_output' in run_df
	return (run_df['original_target'] == run_df['parsed_output']).mean()


def eval_sym_df(run_df):
	assert 'parsed_output' in run_df
	return (run_df['original_target'].apply(expr_to_sympy_w_except) == run_df['parsed_output']).mean()


def expr_to_sympy_w_except(expr):
	transformations = (standard_transformations + (implicit_multiplication_application,))
	parse_expr_part = functools.partial(parse_expr, transformations=transformations)
	
	try:
		parsed_expr = parse_expr_part(expr)
	except SyntaxError as err:
		print(f"Could not parse symbolic expression: {expr}")
		parsed_expr = parse_expr_part('-100')
	return parsed_expr


if __name__ == '__main__':
	main()
