import torch
from torch.utils.data import Dataset
from tokenizer_utils import load_tokenizer, get_files
import random
import ftfy


class GPTDataset(Dataset):
  def __init__(self, data_path, tokenizer, n_positions=1024):
    self.data_path = data_path
    self.files_path = get_files(data_path)

    self.n_positions = n_positions
    self.tokenizer = tokenizer


  def __len__(self):
    return len(self.files_path)


  def __getitem__(self, idx):
    file = open(self.files_path[idx], "r", encoding='utf8').read()
    
    # Encoding file
    file_encoded = self.tokenizer.encode(file)

    assert len(file_encoded) >= self.n_positions, "All files encoded len must be bigger than n_positions"

    # Adding BOS and EOS tokens
    file_encoded.append(self.tokenizer.eos_token_id)
    file_encoded.insert(0, self.tokenizer.bos_token_id)

    # Sampling random window of the text
    start = random.randint(0, (len(file_encoded) - (self.n_positions + 1)))
    end = start + (self.n_positions + 1)
    file_encoded = file_encoded[start:end]

    file_encoded = torch.LongTensor(file_encoded)
    return file_encoded


if __name__ == "__main__":
  tokenizer = load_tokenizer()

  dataset = GPTDataset('./data/train', tokenizer, n_positions=512)
  sample = dataset[0]
  print(sample)
  print("dataset size:",dataset.__len__())
  print("sample size:",sample.size())
  print(ftfy.fix_text(tokenizer.decode(sample)))

  print("=> Running all samples")
  for i in range(dataset.__len__()):
    if dataset[i].size()[0] != 513:
      print(dataset[i].size())
