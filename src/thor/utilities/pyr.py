import abc

class format_conversion(abc):
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    @abc.abstractmethod
    def convert(self):
        pass
