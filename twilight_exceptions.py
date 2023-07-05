import os.path


class TextFileError(Exception):
    """ A user-defined exception for identifying a
        text file-specific issue """
    def __init__(self, filename, msg=''):
        super().__init__("Text File Error: " + msg + " (" + filename + ")")
        self.filename = filename
        self.msg = msg


