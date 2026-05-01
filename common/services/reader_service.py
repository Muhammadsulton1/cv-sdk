from common.runners.basic_runner import BaseServiceRunner
from common.runners.reader import ReaderProcessor


if __name__ == "__main__":
    BaseServiceRunner(ReaderProcessor).run()
