#!/bin/python

""" Merges a shelf A and a shelf B into a shelf C = A | B """

import argparse
import shelve

from tqdm.autonotebook import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merges a shelf A and a shelf B into a shelf C = A | B')
    parser.add_argument('A', type=str, nargs=1)
    parser.add_argument('B', type=str, nargs=1)
    parser.add_argument('C', type=str, nargs=1)
    args = parser.parse_args()

    with shelve.open(parser.A, 'r') as A, shelve.open(parser.B, 'r') as B, shelve.open(parser.C) as C:
        print(f'Gathering the keys stored in {parser.A}')
        keys_A = A.keys()
        print(f'Gathering the keys stored in {parser.B}')
        keys_B = B.keys()

        for key in tqdm(sorted(keys_A | keys_B), desc=f'Storing {parser.A} | {parser.B} in {parser.C}'):
            C[key] = A[key] if key in A else B[key]
