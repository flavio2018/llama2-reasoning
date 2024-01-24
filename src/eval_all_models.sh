#!/bin/bash

# llama2-7b-chat zero-shot
python eval_llama.py model_name=llama2-7b-chat task_name=listops prompt_type=zero_shot
python eval_llama.py model_name=llama2-7b-chat task_name=arithmetic prompt_type=zero_shot
python eval_llama.py model_name=llama2-7b-chat task_name=algebra prompt_type=zero_shot

# llama2-13b-chat zero-shot
python eval_llama.py model_name=llama2-13b-chat task_name=listops prompt_type=zero_shot
python eval_llama.py model_name=llama2-13b-chat task_name=arithmetic prompt_type=zero_shot
python eval_llama.py model_name=llama2-13b-chat task_name=algebra prompt_type=zero_shot

# llama2-70b-chat zero-shot
# python eval_llama.py model_name=llama2-70b-chat task_name=listops prompt_type=zero_shot
# python eval_llama.py model_name=llama2-70b-chat task_name=arithmetic prompt_type=zero_shot
# python eval_llama.py model_name=llama2-70b-chat task_name=algebra prompt_type=zero_shot

# llama2-7b-chat zero-shot-cot
python eval_llama.py model_name=llama2-7b-chat task_name=listops prompt_type=zero_shot_cot
python eval_llama.py model_name=llama2-7b-chat task_name=arithmetic prompt_type=zero_shot_cot
python eval_llama.py model_name=llama2-7b-chat task_name=algebra prompt_type=zero_shot_cot

# llama2-13b-chat zero-shot-cot
python eval_llama.py model_name=llama2-13b-chat task_name=listops prompt_type=zero_shot_cot
python eval_llama.py model_name=llama2-13b-chat task_name=arithmetic prompt_type=zero_shot_cot
python eval_llama.py model_name=llama2-13b-chat task_name=algebra prompt_type=zero_shot_cot

# llama2-70b-chat zero-shot-cot
python eval_llama.py model_name=llama2-70b-chat task_name=listops prompt_type=zero_shot_cot
python eval_llama.py model_name=llama2-70b-chat task_name=arithmetic prompt_type=zero_shot_cot
python eval_llama.py model_name=llama2-70b-chat task_name=algebra prompt_type=zero_shot_cot

# mammoth-7b
python eval_llama.py model_name=mammoth-7b task_name=listops prompt_type=zs_mammoth
python eval_llama.py model_name=mammoth-7b task_name=arithmetic prompt_type=zs_mammoth
# python eval_llama.py model_name=mammoth-7b task_name=algebra prompt_type=zs_mammoth

# mammoth-13b
python eval_llama.py model_name=mammoth-13b task_name=listops prompt_type=zs_mammoth
python eval_llama.py model_name=mammoth-13b task_name=arithmetic prompt_type=zs_mammoth
python eval_llama.py model_name=mammoth-13b task_name=algebra prompt_type=zs_mammoth

# mammoth-70b
# python eval_llama.py model_name=mammoth-70b task_name=listops prompt_type=zs_mammoth
# python eval_llama.py model_name=mammoth-70b task_name=arithmetic prompt_type=zs_mammoth
# python eval_llama.py model_name=mammoth-70b task_name=algebra prompt_type=zs_mammoth

# metamath-7b
# python eval_llama.py model_name=metamath-7b task_name=listops prompt_type=zs_metamath
# python eval_llama.py model_name=metamath-7b task_name=arithmetic prompt_type=zs_metamath
# python eval_llama.py model_name=metamath-7b task_name=algebra prompt_type=zs_metamath

# metamath-13b
python eval_llama.py model_name=metamath-13b task_name=listops prompt_type=zs_metamath
python eval_llama.py model_name=metamath-13b task_name=arithmetic prompt_type=zs_metamath
python eval_llama.py model_name=metamath-13b task_name=algebra prompt_type=zs_metamath

# metamath-70b
# python eval_llama.py model_name=metamath-70b task_name=listops prompt_type=zs_metamath
# python eval_llama.py model_name=metamath-70b task_name=arithmetic prompt_type=zs_metamath
# python eval_llama.py model_name=metamath-70b task_name=algebra prompt_type=zs_metamath
