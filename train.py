import torch
import torch.nn as nn
import torch.optim as optim
from dataset import GPTDataset
from model_hparams import ModelHParams
from model import GPT
from tokenizer_utils import load_tokenizer, train_tokenizer
from utils import count_parameters, generate_sentence, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import datetime


# Creating the dataset and tokenizer
load_trained_tokenizer = True
data_path = './data/train'
h_params = ModelHParams()

if not load_trained_tokenizer:
  train_tokenizer(data_path, h_params.vocab_size)

tokenizer = load_tokenizer()

train_data = GPTDataset(data_path, tokenizer, n_positions=h_params.n_positions)



# Setup the training phase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = False
save_model = True


# Training Hyperparameters
num_epochs = 100
lr_rate = 6e-4
batch_size = 1


# Tensorboard for nice plots
writer = SummaryWriter("runs/loss_plot")
step = 0


# DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Creating the model
model = GPT(
  vocab_size = h_params.vocab_size, 
  n_layer = h_params.n_layer, 
  n_embd = h_params.n_embd,
  n_head = h_params.n_head,
  n_positions = h_params.n_positions,
  attn_pdrop = h_params.attn_pdrop,
  embd_pdrop = h_params.embd_pdrop,
  resid_pdrop = h_params.resid_pdrop,
  layer_norm_epsilon = h_params.layer_norm_epsilon)

print("GPT Model parameters:", count_parameters(model))

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.95), weight_decay=0.1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=(6e-4*0.1))

criterion = nn.CrossEntropyLoss()


if load_model:
  load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


sentence = "Escreva uma notícia sobre a situação econômica atual do Brasil:"
losses = []
batch_idxs = []

# Training the model
for epoch in range(num_epochs):
  print(f"Epoch [{epoch} / {num_epochs}]")

  if save_model:
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)


  # Evaluating the model
  model.eval()
  generate_sentence(model, sentence, tokenizer, device, steps=50, temperature=3.0, sample=True, top_k=40)

  now = datetime.datetime.now()
  print("Time now:", now.time())

  # Training the model
  model.train()
  for batch_idx, batch in enumerate(train_loader):
    inp_data = batch.to(device) # (batch_size, n_tokens)

    # Forward prop
    output = model(inp_data[:, :-1]) # (batch_size, n_tokens, vocab_size)

    output = output.reshape(-1, output.shape[2])
    target = inp_data[:, 1:].reshape(-1)

    optimizer.zero_grad()

    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

    optimizer.step()
    losses.append(loss.item())
    batch_idxs.append(batch_idx)

    # try:
    #   writer.add_scalar("Training loss", loss, global_step=step)
    # except Exception as e:
    #   print(e)

    step+=1

  scheduler.step()

  df = pd.DataFrame()
  df["Loss"] = losses
  df["batch_ids"] = batch_idxs
  df.to_csv('./runs/loss.csv')