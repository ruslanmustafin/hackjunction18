import argparse
import csv

import tqdm

# CSV structure:
# 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# 1 - the id of the tweet (2087)
# 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
# 4 - the user that tweeted (robotickilldozr)
# 5 - the text of the tweet (Lyx is cool)


# TSV structure:
# 0 - label
# 1 - text


def main(args):
    
    test_l = 10
    l_id = 0

    pol_to_label = {
        '0': '0', # negative
        '2': '1', # neutral
        '4': '2'  # positive
    }

    out_path = args.o

    if not out_path:
        out_path = '.'.join(args.i.split('.')[:-1] + ['tsv'])

    with open(args.i, encoding='latin-1') as in_file, open(out_path, 'w+') as out_file:
        reader = csv.reader(in_file)

        # in_csv = pd.read_csv(args.i, index_col=False, encoding='latin-1')
        # print(in_csv)

        out_file.writelines(['label\ttext\n'])

        for line in tqdm.tqdm(reader):
            l_id += 1
            pol, _, _, _, _, text = [x.strip() for x in line]
            label = pol_to_label[pol]
            line_out = '\t'.join([label] + list(map(lambda x: f'"{x}"', [text])))
            out_file.write(f'{line_out}\n')

            # if l_id >= test_l:
            #     break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, default=None)

    args = parser.parse_args()



    main(args)