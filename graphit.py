import csv
import sys

import matplotlib.pyplot as plt


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    x = []
    y = []
    header = None

    total_len =file_len(input_filepath)
    # sample_rate = total_len
    sample_rate = total_len / 10000

    with open(input_filepath, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for (index, row) in enumerate(plots):
            if index == 0:
                header = row
            else:
            # elif (index % sample_rate == 0):
                x.append(int(row[0]))
                y.append(float(row[1]))

    plt.figure(figsize=(10, 7))  # 10" x 10"
    plt.plot(x,y, 'ko', label='Recent Average Error %', linestyle='None', markersize=1)
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    plt.title('Net recent average error over course of training')
    plt.legend()
    print('plotting...')
    plt.show()
    # plt.savefig(output_filepath)
    # print("Wrote: {}".format(output_filepath))

if __name__ == '__main__':
    main()
