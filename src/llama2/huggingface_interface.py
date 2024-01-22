import huggingface_hub
import tqdm
import pickle
import datetime as dt


class HFInterface:

    def __init__(self):
        self.client = huggingface_hub.InferenceClient(model="http://127.0.0.1:8000")
        self.default_system_message = "You are a helpful assistant."
        self.max_new_tokens = 4000
        self.max_new_tokens_decrease = 10
        self.max_new_tokens_0shot_cot = 500
        self.max_new_tokens_0shot_cot_decrease = 1

    @staticmethod
    def _chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    @staticmethod
    def _format_system_message(msg):
        return f"<<SYS>>\n{msg}\n<</SYS>>"

    @staticmethod
    def _format_instruction(msg):
        return f"[INST] {msg} [/INST]"

    def _format_conversation_turn(self, instruction, answer, sys=None):
        if sys is None:
            formatted_instruction = self._format_instruction(instruction)
            return f"<s>{formatted_instruction} {answer} </s>"
        else:
            sys_and_instruction = self._format_system_message(sys) + '\n\n' + instruction
            formatted_instruction = self._format_instruction(sys_and_instruction)
            return f"<s>{formatted_instruction} {answer} </s>"

    def _format_instruction_prompt(self, instruction, sys=None):
        if sys is None:
            formatted_instruction = self._format_instruction(instruction)
            return f"<s>{formatted_instruction}"
        else:
            sys_and_instruction = self._format_system_message(sys) + '\n\n' + instruction
            formatted_instruction = self._format_instruction(sys_and_instruction)
            return f"<s>{formatted_instruction}"
        
    def _build_structured_prompts_simple(self, prompts, system=None):
        return [self._format_instruction_prompt(prompt, system) for prompt in prompts]

    def _build_structured_prompts_conversation(self, conversation_prompts, system=None):
        structured_prompts = []
        for conversation in conversation_prompts:
            structured_prompt = ""
            for turn_idx, turn in enumerate(self._chunker(conversation, 2)):
                if len(turn) == 1:
                    instruction = turn[0].strip()
                    structured_prompt += self._format_instruction_prompt(instruction)
                elif turn_idx == 0:
                    instruction, answer = turn[0].strip(), turn[1]
                    structured_prompt += self._format_conversation_turn(instruction, answer, system)
                else:
                    instruction, answer = turn[0].strip(), turn[1]
                    structured_prompt += self._format_conversation_turn(instruction, answer)
            structured_prompts.append(structured_prompt)
        return structured_prompts

    def _build_structured_prompts(self, prompts, system=None):
        assert isinstance(prompts, list)

        if isinstance(prompts[0], list):
            return self._build_structured_prompts_conversation(prompts, system)
        elif isinstance(prompts[0], str):
            return self._build_structured_prompts_simple(prompts, system)
        else:
            assert False, f"Wrong prompt type {type(prompts[0])}"

    def query_model(self, prompts, system=None):
        print(f"Querying model with {len(prompts)} prompts...")
        outputs = []
        
        if system is None:
            system = self.default_system_message
        
        structured_prompts = self._build_structured_prompts(prompts, system)
        
        for prompt in structured_prompts:
            max_new_tokens = self.max_new_tokens
            keep_querying = True
            
            while keep_querying:
                try:
                    text_generation_res = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, details=True)
                
                except huggingface_hub.inference._text_generation.ValidationError as err:
                    max_new_tokens -= self.max_new_tokens_decrease

                else:
                    outputs.append(text_generation_res.generated_text)
                    keep_querying = False
                
                    # if text_generation_res.details['finish_reason'] == 'length':
                    #     outputs[-1] = '[L]' + outputs[-1]
        
        return outputs

    def query_model_zero_shot_cot(self, prompts, system=None):
        print(f"Querying model with {2*len(prompts)} prompts...")
        self.zero_cot_first_outputs = []

        if system is None:
            system = self.default_system_message
        
        structured_prompts = self._build_structured_prompts(prompts, system)
        
        for prompt in structured_prompts:
            max_new_tokens = self.max_new_tokens
            keep_querying = True

            while keep_querying:
                try:
                    text_generation_res = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, details=True)
                
                except huggingface_hub.inference._text_generation.ValidationError as err:
                    max_new_tokens -= self.max_new_tokens_decrease

                else:
                    self.zero_cot_first_outputs.append(text_generation_res.generated_text)
                    keep_querying = False

        prompts_with_answer = [[prompt, output, "Thank you, now just output the final answer without writing anything else."] for prompt, output in zip(prompts, self.zero_cot_first_outputs)]

        structured_prompts_with_answer = self._build_structured_prompts(prompts_with_answer, system)
        final_outputs = []

        for prompt in structured_prompts_with_answer:
            max_new_tokens = self.max_new_tokens_0shot_cot
            keep_querying = True

            while keep_querying:
                try:
                    text_generation_res = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, details=True)
                
                except huggingface_hub.inference._text_generation.ValidationError as err:
                    max_new_tokens -= self.max_new_tokens_0shot_cot_decrease

                else:
                    final_outputs.append(text_generation_res.generated_text)
                    keep_querying = False

                    # if text_generation_res.details['finish_reason'] == 'length':
                    #     final_outputs[-1] = '[L]' + final_outputs[-1]

        return final_outputs
