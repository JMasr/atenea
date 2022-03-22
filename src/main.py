from model import Pipeline
from argparser import parse_arguments

from timeit import default_timer as timer
from datetime import timedelta

if __name__ == '__main__':
    start = timer()  # Start timer

    args = parse_arguments()
    rich_t = Pipeline(args)
    rich_t.run_models()
    rich_t.make_srt()

    end = timer()  # End timer
    print(timedelta(seconds=end - start))
