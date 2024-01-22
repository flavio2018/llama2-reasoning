#!/usr/bin/env python


import hydra
from llama2.querier import ModelQuerierOnTask


@hydra.main(config_path="../conf/", config_name="query_llama", version_base='1.2')
def main(cfg):
	querier = ModelQuerierOnTask(cfg)
	querier.query_model()


if __name__ == '__main__':
	main()
