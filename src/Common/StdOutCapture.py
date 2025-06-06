import sys
import io

class StdOutCapture(io.StringIO):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def write(self, data):
        try:
            sys.__stdout__.write(data)  # Write to the original stdout (console)
            self.file.write(data)  # Write to the file
        except:
            None
            
    def flush(self):
        sys.__stdout__.flush()
        try:
            self.file.flush()
        except:
            None