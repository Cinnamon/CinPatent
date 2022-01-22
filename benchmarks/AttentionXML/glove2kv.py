from argparse import ArgumentParser
from gensim.models.keyedvectors import KeyedVectors

parser = ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, default='glove.840B.300d.txt', help='GloVe file path')
parser.add_argument('-o', '--output_file', type=str, default='glove.840B.300d.kv', help='Output KeyVectored file')
args = parser.parse_args()

print(f'Reading GloVe from {args.input_file}...')
kv = KeyedVectors.load_word2vec_format(args.input_file, binary=False, no_header=True)
num_lines, num_dims = len(kv), kv.vector_size
print('Embedding size: {}x{}'.format(num_lines, num_dims))
print(f'Wring GloVe KeyVectored to {args.output_file}...')
kv.save(args.output_file)