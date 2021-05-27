import argparse

from frameworks.Triplet.finetune import triplet_finetune
from frameworks.Triplet.extract_features import extract
from frameworks.DeepRobustClustering.finetune import drc_finetune
from frameworks.Unsupervised.finetune import unsupervised_finetune


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning cuBERT')
    parser.add_argument('-f', '--framework', dest='framework', type = str.lower, required=True,
                        help='The framework for fine-tuning cuBERT. Valid options are:\nTriplet\nDRC\nUnsupervised')

    args = parser.parse_args()
    if args.framework == 'triplet':
        # Fine-tune model
        triplet_finetune()
        # Extract features from fine-tuned model
        extract()
    elif args.framework == 'drc':
        drc_finetune()
    elif args.framework == 'unsupervised':
        unsupervised_finetune()
    else:
        print("The chosen framework is not supported. Valid options are:\nTriplet\nDRC\nUnsupervised")


if __name__ == "__main__":
    main()