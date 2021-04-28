import config
from transformers import BertForPreTraining, BertConfig

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, output_path):
    print("Converting weights...")
    model_config = BertConfig.from_json_file(bert_config_file)
    model = BertForPreTraining.from_pretrained(pretrained_model_name_or_path=tf_checkpoint_path, from_tf=True, config=model_config)
    model.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.bert_config_file,
                                     args.pytorch_dump_path)