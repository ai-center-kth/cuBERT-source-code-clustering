from .string_transformer import String_Transformer


class File_Transformer(String_Transformer):
    """Reads a file and uses the string transformer to transform the source
    code. Saves the transformed source code to a new file.
        Parameters:
            transformations_to_perform: list of functions
        Returns:
            None, writes to outfile.
    """

    def __init__(self, transformations_to_perform):
        super().__init__(transformations_to_perform)

    def _read_file(self, file_path):
        try:
            with open(file_path, "r") as f:
                code = f.read()
        except IOError as e:
            print(f"Error Opening File: {file_path}")
            raise(e)
        except Exception as e:
            print(f"Unexpected Error on File: {file_path}")
            raise(e)
        f.close()
        return code

    def _write_to_file(self, file_path, code_string):
        try:
            with open(file_path, "w") as f:
                f.write(code_string)
        except IOError as e:
            print(f"Error Opening File: {file_path}")
            raise(e)
        except Exception as e:
            print(f"Unexpected Error on File: {file_path}")
            raise(e)
        f.close()
        
    def transform(self, in_file, out_file):
        """
        Reads in a source code file, transforms it, and outputs result to
        a file.
            Parameters:
                in_file: file path to read source code from.
                out_file: file path to write transformed source code to.
        """
        src_code_str = self._read_file(in_file)
        new_source_str = String_Transformer.transform(self, src_code_str)
        self._write_to_file(out_file, new_source_str)
