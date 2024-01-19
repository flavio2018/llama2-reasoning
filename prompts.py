import random
from gpt.examples import build_examples


class PromptBuilder:

    def __init__(self):
        self.task_name = None
        self.blueprints = None
        self.prompts_wout_examples = ["zero_shot", "zero_shot_cot"]
        self.prompts_with_examples = ["cot_equation", "cot_verbal", "few_shot"]
        self.verbal_examples_start_solution = [
            "Let's solve the following expression: {}.\n",
            "We need to solve the following expression: {}.\n",
            "The expression we need to solve is: {}.\n",
            "Let us recall the expression to be solved: {}.\n",
        ]
        self.verbal_examples_fillers_middle = [
            "Simplifying an expression without nested parentheses, we get: {}.\n",
            "Taking an immediate solution step, we obtain: {}.\n",
            "By solving a simple expression, we obtain: {}.\n",
            "Simplifying the expression, it becomes: {}\n",
            "Solving a expression within a single pair of brackets, we get: {}.\n",
        ]
        self.verbal_examples_fillers_end = [
            "As this expression is in a simple form, we can get to the final result: {}\n",
            "Simplifying the expression, we get to the final result: {}\n",
            "Taking an immediate solution step, we get to the final result: {}.\n",
        ]
        self.verbal_examples_fillers_end_factorization = [
            "As this expression is in a simple form, we can get to the final result factoring by grouping: {}\n",
            "Simplifying the expression and factoring by grouping, we get to the final result: {}\n",
            "Taking an immediate solution step and factoring by grouping, we get to the final result: {}.\n",
        ]
    
    def _build_prompt_no_examples(self, batch, prompt_type):
        return [self.blueprints[prompt_type].format(sample) for sample in batch]

    def _build_prompt_cot_equation(self, batch, examples):
        prompts_cot_equation = []
        for sample in batch:
            conversation = []
            for steps in examples:
                conversation.append(self.blueprints["examples"]['question'].format(steps[0]))
                conversation.append(self.blueprints["examples"]['answer'].format("=\n".join(steps) + "."))
            conversation.append(self.blueprints["examples"]['question'].format(sample))
            prompts_cot_equation.append(conversation)
        return prompts_cot_equation

    def _build_prompt_cot_verbal(self, batch, examples):
        prompts_cot_verbal = []
        for sample in batch:
            conversation = []
            for steps in examples:
                conversation.append(self.blueprints["examples"]["question"].format(steps[0]))
                verbal_solution_exemplar = ''.join([random.choice(self.verbal_examples_start_solution).format(steps[0])])
                verbal_solution_exemplar += ''.join([random.choice(self.verbal_examples_fillers_middle).format(step) for step in steps[1:-1]]) 
                if steps[-1].count('(') > 0:
                    verbal_solution_exemplar += ''.join([random.choice(self.verbal_examples_fillers_end_factorization).format(steps[-1])])
                else:
                    verbal_solution_exemplar += ''.join([random.choice(self.verbal_examples_fillers_end).format(steps[-1])])
                conversation.append(verbal_solution_exemplar)
            conversation.append(self.blueprints['examples']['question'].format(sample))
            prompts_cot_verbal.append(conversation)
        return prompts_cot_verbal

    def _build_prompt_few_shot(self, batch, examples):
        prompts_few_shot = []
        for sample in batch:
            conversation = []
            for steps in examples:
                conversation.append(self.blueprints['examples']['question'].format(steps[0]))
                conversation.append(self.blueprints['examples']['answer'].format(f"{steps[0]}={steps[-1]}."))
            conversation.append(self.blueprints['examples']['question'].format(sample))
            prompts_few_shot.append(conversation)
        return prompts_few_shot

    def _build_prompt_with_examples(self, batch, prompt_type, examples):
        if prompt_type == 'few_shot':
            return self._build_prompt_few_shot(batch, examples)
        elif prompt_type == 'cot_equation':
            return self._build_prompt_cot_equation(batch, examples)
        elif prompt_type == 'cot_verbal':
            return self._build_prompt_cot_verbal(batch, examples)

    def _build_examples(self, prompt_type):
        if prompt_type in self.prompts_wout_examples:
            return None
        elif prompt_type in self.prompts_with_examples:
            return build_examples(self.task_name)
        else:
            assert False, f"Wrong prompt type: {prompt_type}."

    def build_prompt(self, batch, prompt_type):
        if prompt_type == 'self_consistency':
            prompt_type = 'zero_shot_cot'

        examples = self._build_examples(prompt_type)

        if examples is None:
            return self._build_prompt_no_examples(batch, prompt_type)
        else:
            return self._build_prompt_with_examples(batch, prompt_type, examples)


class ArithmeticPromptBuilder(PromptBuilder):

    def __init__(self):
        super().__init__()
        self.task_name = 'arithmetic'
        self.blueprints = {
            "zero_shot": "Q: Solve the following arithmetic expression computing the modulo 100 of each intermediate value "
                         "if it's positive, and the modulo -100 if it's negative:\n{}.\n\n"
                         "A: The final result is (arabic numerals):",

            "zero_shot_cot": "Q: Solve the following arithmetic expression computing the modulo 100 of each intermediate value "
                             "if it's positive, and the modulo -100 if it's negative:\n{}.\n\n"
                             "A: Let's think step-by-step.",

            "examples": {"question": "Solve the following arithmetic expression taking each intermediate value modulo 100 if it's positive, and modulo -100 if it's negative: {}.",
                         "answer": "{}"},
        }


class AlgebraPromptBuilder(PromptBuilder):
        
    def __init__(self):
        super().__init__()
        self.task_name = 'algebra'
        self.blueprints = {
            "zero_shot": "Q: Simplify the following algebraic expression, computing the modulo 100 of the numerical coefficient of each "
                         "intermediate value if it's positive, and the modulo -100 if it's negative:\n{}.\n"
                         "If possible, factor by grouping the final result.\n\n"
                         "A: The final result is (algebraic expression):",

            "zero_shot_cot": "Q: Simplify the following algebraic expression, computing the modulo 100 of the numerical coefficient of each "
                             "intermediate value if it's positive, and the modulo -100 if it's negative:\n{}.\n"
                             "If possible, factor by grouping the final result.\n\n"
                             "A: Let's think step-by-step.",

            "examples": {"question": "Solve the following algebraic expression taking the numerical coefficient of each intermediate value "
                                     "modulo 100 if it's positive, and modulo -100 if it's negative:\n{}.\n"
                                     "If possible, factor by grouping the final result.",
                         "answer": "{}"},
        }


class ListopsPromptBuilder(PromptBuilder):

    def __init__(self):
        super().__init__()
        self.task_name = 'listops'
        self.blueprints = {
            "zero_shot": "Q: MIN, MAX and SM are operators on lists of single-digit integers which have the semantics of "
                         "minimum, maximum and sum modulo 10, respectively. "
                         "Solve the following expression involving these operators:\n{}.\n\n"
                         "A: The final result is (arabic numeral):",

            "zero_shot_cot": "Q: MIN, MAX and SM are operators on lists of single-digit integers which have the semantics of "
                             "minimum, maximum and sum modulo 10, respectively. "
                             "Solve the following expression involving these operators:\n{}.\n\n"
                             "A: Let's think step-by-step.",

            "examples": {"question": "MIN, MAX and SM are operators on lists of single-digit integers which have the semantics of "
                                     "minimum, maximum and sum modulo 10, respectively. "
                                     "Solve the following expression involving these operators:\n{}.",
                         "answer": "{}"},
        }


def get_prompt_builder(task_name):
    if task_name == 'listops':
        return ListopsPromptBuilder()
    elif task_name == 'arithmetic':
        return ArithmeticPromptBuilder()
    elif task_name == 'algebra':
        return AlgebraPromptBuilder()
    else:
        assert False, f"Wrong task name: {task_name}"
