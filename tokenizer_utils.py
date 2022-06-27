from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import os


# Getting files to train tokenizer
def get_files(path):
  files = os.listdir(path)
  files_path = []

  for file in files:
    files_path.append(os.path.join(path, file))

  return files_path



def train_tokenizer(data_path, vocab_size=50_257, min_frequency=2):
  tokenizer = ByteLevelBPETokenizer()

  files = get_files(data_path)

  print("=> Training tokenizer")
  tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
    "<sos>",
    "<pad>",
    "<eos>",
    "<unk>",
    "<mask>"
  ])

  print("=> Saving tokenizer")
  tokenizer.save("tokenizer.json")



def load_tokenizer():
  print("=> Loading tokenizer")
  tokenizer = PreTrainedTokenizerFast.from_pretrained('./tokenizer.json', use_auth_token=True, local_files_only=True)
  tokenizer.add_special_tokens({'pad_token': '<pad>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'bos_token': '<eos>'})
  return tokenizer



if __name__ == '__main__':
  train_tokenizer('./data/train')