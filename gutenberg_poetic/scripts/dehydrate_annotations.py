import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reformat annotated lines to a minimal form with line references only."
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to the input data.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where the output will be saved'
    )
    return parser.parse_args()


def main(args):

    with open(args.input_path) as f:
        lines = f.readlines()

    with open(args.output_path, 'w') as f:
        for line in lines:
            label, _, location = line.split('\t')[:3]
            gd_num = location.split('_')[0].split('.')[1]
            line_num = location.split('.')[-1]
            f.write('\t'.join([label, gd_num, line_num]) + '\n')


if __name__ == "__main__":
    main(parse_args())
