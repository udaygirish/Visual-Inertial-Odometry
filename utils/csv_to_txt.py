# Convert CSV file to TXT file

import csv
import argparse

def csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        with open(txt_file, 'w') as txt:
            for row in reader:
                txt.write(' '.join(row) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CSV file to TXT file')
    parser.add_argument('--csv', type=str, help='Input CSV file')
    parser.add_argument('--txt', type=str, help='Output TXT file')

    args = parser.parse_args()
    csv_to_txt(args.csv, args.txt)
