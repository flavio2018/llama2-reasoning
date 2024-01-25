import pandas as pd
import re


def load_test_df(task_name):
	df = pd.read_csv(f'../data/{task_name}/test.csv')

	if task_name == 'listops':
		df['X'] = df['X'].apply(reformat_listops_expression)
	
	return df


def reformat_listops_expression(expr):
	breakpoint()
	listops_re = re.compile(r'(\d)|(SM|MIN|MAX)|([\[\]])|([?.#$])')
	matches = listops_re.findall(expr)
	expr_w_spaces = " ".join([[submatch for submatch in match if submatch][0] for match in matches])
	return expr_w_spaces.replace('[ ', '[').replace(' ]', ']').replace('MIN ','MIN(').replace('MAX ','MAX(').replace('SM ','SM(').replace(' ', ',').replace('[', '').replace(']', ')')
