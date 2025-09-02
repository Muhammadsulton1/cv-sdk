from src.data_reader import ReaderManager
from src.basic_runner import BasicRunner


if __name__ == '__main__':
    BasicRunner().run_process(processors=[ReaderManager()])
