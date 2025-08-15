from src.basic_runner import BasicRunner
from src.router import RouterManager

if __name__ == '__main__':
    BasicRunner().run_process(processors=[RouterManager()])

