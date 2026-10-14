"""Create a tiny synthetic checkpoint for tests, never a benchmark artifact."""
import argparse
from pathlib import Path
import sys


def create_fixture(path, repository, tied=False):
    import torch
    sys.path.insert(0, str(repository))
    from Models.modeling import BertConfig, BertForSequenceClassification
    torch.manual_seed(7)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'a', 'good', 'movie', 'bad']
    config = BertConfig(len(vocabulary))
    config.hidden_size = 8
    config.intermediate_size = 16
    config.num_attention_heads = 2
    config.num_hidden_layers = 1
    config.hidden_act = 'relu'
    config.layer_norm = 'no_var'
    config.max_position_embeddings = 32
    config.hidden_dropout_prob = config.attention_probs_dropout_prob = 0
    model = BertForSequenceClassification(config, 2)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([0., 0.] if tied else [2., -2.]))
    torch.save(model.state_dict(), path / 'pytorch_model.bin')
    config.to_json_file(str(path / 'bert_config.json'))
    (path / 'vocab.txt').write_text('\n'.join(vocabulary) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tied', action='store_true')
    args = parser.parse_args()
    create_fixture(args.output, args.repository, args.tied)
