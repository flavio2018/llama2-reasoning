import re
import warnings
import sys


class Llama2OutputParser:

    def __init__(self):
        self.output_type = None
        self.error_counter = 0

    def _filter_matches(self, matches):
        if isinstance(matches[0], tuple):
            matches = [(m[0].strip(), m[1].strip(), m[2].strip()) for m in matches]
            matches = [match for match in matches if match != ('', '', '')]
        return matches

    def _simple_parse_outputs(self, outputs):
        outputs = self._preprocessing_step(outputs)
        try:
            matches = self.simple_output_re.findall(outputs)
            matches = self._filter_matches(matches)
            match = matches[-1]
        except IndexError:
            warnings.warn(f"Match is empty for Llama2 output: {outputs}")
            match = '-100'
            self.error_counter += 1

        if isinstance(match, tuple):
            match = match[0]
            if match == '':
                warnings.warn(f"Match is empty for Llama2 output: {outputs}")
                match = '-100'
                self.error_counter += 1

        if self.output_type == int:
            int_match = int(match)
            if (int_match > sys.float_info.max) or (int_match < -sys.float_info.max):
                print(f'Integer match {int_match} out of float type range.')
                return -100
            return int_match
        elif self.output_type == str:
            return match
        
    def parse_outputs(self, outputs):
        return self._simple_parse_outputs(outputs)

class Llama2ListopsOutputParser(Llama2OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r'\d')
        self.output_type = int
        self.task_name = 'listops'

    def _preprocessing_step(self, output):
        output = output.replace('modulo 10', '')
        output = output.replace('mod 10', '')
        output = output.replace('Modulo 10', '')
        output = output.replace('Mod 10', '')
        output = output.replace('modulus 10', '')
        output = output.replace('Modulus 10', '')
        return output

class Llama2ArithmeticOutputParser(Llama2OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r'\-{0,1}\d\d*')
        self.output_type = int
        self.task_name = 'arithmetic'

    def _preprocessing_step(self, output):
        output = output.replace('modulo -100', '')
        output = output.replace('Modulo -100', '')
        output = output.replace('(mod -100)', '')
        output = output.replace('mod -100', '')
        output = output.replace('Mod -100', '')
        output = output.replace('modulo 100', '')
        output = output.replace('Modulo 100', '')
        output = output.replace('(mod 100)', '')
        output = output.replace('mod 100', '')
        output = output.replace('Mod 100', '')
        output = output.replace('modulus 100', '')
        output = output.replace('Modulus 100', '')
        return output

class Llama2AlgebraOutputParser(Llama2OutputParser):

    def __init__(self):
        super().__init__()
        # self.simple_output_re = re.compile(r'([+-]*[0-9]*[0-9]*[* ]*[abxy*]*[ ]*([(]([-+0-9]|[-abxy])+)*[abxy*]*[ ]*[+-]*[ ]*[0-9]*[* ]*[abxy* ]*[/0-9]*[)]*[abxy*]*)')
        self.simple_output_re = re.compile(r'([+-]{0,1}\d+[abxy*]+|[+-]*0+|[+-]{0,1}[abxy*]+)')
        self.output_type = str
        self.task_name = 'algebra'

    def _preprocessing_step(self, output):
        output = output.replace('modulo -100', '')
        output = output.replace('Modulo -100', '')
        output = output.replace('(mod -100)', '')
        output = output.replace('mod -100', '')
        output = output.replace('Mod -100', '')
        output = output.replace('modulo 100', '')
        output = output.replace('Modulo 100', '')
        output = output.replace('(mod 100)', '')
        output = output.replace('mod 100', '')
        output = output.replace('Mod 100', '')
        output = output.replace('modulus 100', '')
        output = output.replace('Modulus 100', '')
        return output
        
def build_parser(task_name):
    if task_name == 'algebra':
        return Llama2AlgebraOutputParser()
    elif task_name == 'arithmetic':
        return Llama2ArithmeticOutputParser()
    elif task_name == 'listops':
        return Llama2ListopsOutputParser()
    else:
        assert False, f"Wrong task name {task_name}"
