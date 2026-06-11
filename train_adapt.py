from trainer_adapt import AdaptTrainer
from options import MonodepthOptions
import sys

options = MonodepthOptions()


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == "__main__":
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opts = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opts = options.parser.parse_args()
    trainer = AdaptTrainer(opts)
    trainer.train()
