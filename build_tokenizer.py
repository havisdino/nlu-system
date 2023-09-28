import sentencepiece as spm
import jsonlines as jsl
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default=None)
    parser.add_argument('-s', '--size', type=int, default=768)
    
    args = parser.parse_args()
    assert args.file is not None
    
    with (jsl.open(args.file) as source,
          open('sentences.txt', 'w') as destination):
        for sample in source:
            sentence = sample['sentence'].lower()
            if not sentence.endswith('\n'):
                sentence += '\n'
            destination.write(sentence)
    
    spm.SentencePieceTrainer.train(
        input='sentences.txt',
        model_prefix='tokenizer/tok',
        vocab_size=args.size,
        user_defined_symbols=['▁[', '▁:', '▁]']
    )
    os.remove('sentences.txt')


if __name__ == '__main__':
    main()
