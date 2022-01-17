from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description = "Model trainer")
    parser.add_argument("--ckpt_directory", type=str, default='')
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--train_data_path", type = str, default='../data/Reuters/train.json')
    parser.add_argument("--val_data_path", type = str, default='../data/Reuters/val.json')
    parser.add_argument("--test_data_path", type = str, default='../data/Reuters/test.json')
    parser.add_argument("--device", type = str, default='cuda')
    parser.add_argument("--pretrained", type = str, default='roberta-base',
                        help= "Pretrained Transformer Model")
    parser.add_argument("--checkpoint", type = str, default='')
    parser.add_argument("--tokenizer_name", type = str, default='roberta-base')
    parser.add_argument("--sbert", type = str, default='paraphrase-distilroberta-base-v1')
    parser.add_argument("--mode", type = str, default='train',
                        help= "train/test")
    args = parser.parse_args()
    
    print('args: ', args.threshold)