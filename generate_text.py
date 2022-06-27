import torch
import torch.optim as optim
from utils import count_parameters, generate_sentence, load_checkpoint
from tokenizer_utils import load_tokenizer
from model import GPT
from model_hparams import ModelHParams

device = torch.device("cuda")

h_params = ModelHParams()

model = GPT(
  vocab_size = h_params.vocab_size, 
  n_layer = h_params.n_layer, 
  n_embd = h_params.n_embd,
  n_head = h_params.n_head,
  n_positions = h_params.n_positions,
  attn_pdrop = h_params.attn_pdrop,
  embd_pdrop = h_params.embd_pdrop,
  resid_pdrop = h_params.resid_pdrop,
  layer_norm_epsilon = h_params.layer_norm_epsilon).to(device)

lr_rate = 6e-4

optimizer = optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.95), weight_decay=0.1)

load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

print(count_parameters(model))

tokenizer = load_tokenizer()


model.eval()

while True:
  prompt = input('Prompt: ')
  generate_sentence(model, prompt, tokenizer, device, steps=50, temperature=3, sample=True, top_k=40)