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

    with shelve.open(args.A[0], 'r') as A, shelve.open(args.B[0], 'r') as B, shelve.open(args.C[0]) as C:
        print(f'Gathering the keys stored in {args.A[0]}')
        keys_A = set(A.keys())
        print(f'Gathering the keys stored in {args.B[0]}')
        keys_B = set(B.keys())

        for key in tqdm(sorted(keys_A | keys_B), desc=f'Storing {args.A[0]} | {args.B[0]} in {args.C[0]}'):
            C[key] = B[key] if key in B else A[key]
