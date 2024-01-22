from gpt.prompts import PromptBuilder as GPTPromptBuilder, \
    ListopsPromptBuilder as GPTListopsPromptBuilder, \
    ArithmeticPromptBuilder as GPTArithmeticPromptBuilder, \
    AlgebraPromptBuilder as GPTAlgebraPromptBuilder


class PromptBuilder(GPTPromptBuilder):

    def __init__(self):
        super().__init__()
        self.prompts_wout_examples = ["zero_shot", "zero_shot_role", "zero_shot_cot"]
        
    def _build_prompt_with_examples(self, batch, prompt_type, examples):
        if prompt_type == 'few_shot':
            conversation_batch = self._build_prompt_few_shot(batch, examples)
            return self._conversation_to_llama2(conversation_batch)
    
        elif prompt_type == 'cot_equation':
            conversation_batch = self._build_prompt_cot_equation(batch, examples)
            return self._conversation_to_llama2(conversation_batch)

        elif prompt_type == 'cot_verbal':
            conversation_batch = self._build_prompt_cot_verbal(batch, examples)
            return self._conversation_to_llama2(conversation_batch)

    def _conversation_to_llama2(self, conversation_batch):
        prompts_few_shot = []
        for conversation in conversation_batch:
            qa_prefixes = ["Q: ", "A: "]*3
            exemplars = [''.join(qa) for qa in zip(qa_prefixes, conversation)]
            prompt_few_shot = '\n'.join(exemplars)
            prompt_few_shot = prompt_few_shot + "\nQ: " + conversation[-1]
            prompts_few_shot.append(prompt_few_shot)
        return prompts_few_shot


class ArithmeticPromptBuilder(PromptBuilder, GPTArithmeticPromptBuilder):

    def __init__(self):
        super().__init__()
        self.blueprints["zero_shot_role"] = "You are a brilliant mathematician.\n\n" \
                                            "Q: Solve the following arithmetic expression computing the modulo 100 of each intermediate value " \
                                            "if it's positive, and the modulo -100 if it's negative:\n{}.\n\n" \
                                            "A: The final result is (arabic numerals):"


class AlgebraPromptBuilder(PromptBuilder, GPTAlgebraPromptBuilder):
        
    def __init__(self):
        super().__init__()
        self.blueprints["zero_shot_role"] = "You are a brilliant mathematician.\n\n" \
                                            "Q: Simplify the following algebraic expression, computing the modulo 100 of the numerical coefficient of each " \
                                            "intermediate value if it's positive, and the modulo -100 if it's negative:\n{}.\n" \
                                            "If possible, factor by grouping the final result.\n\n" \
                                            "A: The final result is (algebraic expression):"


class ListopsPromptBuilder(PromptBuilder, GPTListopsPromptBuilder):

    def __init__(self):
        super().__init__()
        self.blueprints["zero_shot_role"] = "You are a brilliant mathematician.\n\n" \
                                            "Q: MIN, MAX and SM are operators on lists of single-digit integers which have the semantics of " \
                                            "minimum, maximum and sum modulo 10, respectively. " \
                                            "Solve the following expression involving these operators:\n{}.\n\n" \
                                            "A: The final result is (arabic numeral):"

def get_prompt_builder(task_name):
    if task_name == 'listops':
        return ListopsPromptBuilder()
    elif task_name == 'arithmetic':
        return ArithmeticPromptBuilder()
    elif task_name == 'algebra':
        return AlgebraPromptBuilder()
    else:
        assert False, f"Wrong task name: {task_name}"
