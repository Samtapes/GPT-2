import torch
from model import GPT
from model_hparams import ModelHParams
from tokenizer_utils import load_tokenizer
import ftfy


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_token(model, token_ixs, device, temperature=1.0, sample=True, top_k=10):
  """Generate a single token giving previous tokens.
  
  Parameters
  ----------
  model : GPT

  token_ixs : list
    List of conditional tokens ids.

  temperature : float
    The higher the more variability and vice versa.

  sample : bool
    If True, we sample from the distribution (=there is randomness). If
    False, we just take the argmax (=there is no randomness).

  top_k : int or None
    If not None then we modify the distribution to only contain the `top_k`
    most probable outcomes.

  Returns
  -------
  new_token_ix : int
    Index of the new token.
  """
  context_token_ixs = token_ixs[-model.n_positions:]
  ixs = torch.tensor(context_token_ixs).to(dtype=torch.long)[
    None, :
  ].to(device) # (1, n_tokens)

  with torch.no_grad():
    logits_all = model(ixs) # (1, n_tokens, vocab_size)
    logits = logits_all[0, -1, :] # (vocab_size,)
    logits = logits / temperature

  if top_k is not None:
    # Find the top k biggest elements, set the remaining elements to -inf
    top_value, _ = torch.topk(logits, top_k) # (top_k)
    logits[logits < top_value.min()] = -float('inf')

  probs = torch.softmax(logits, dim=0) # (vocab_size,)

  if sample:
    new_token_ix = torch.multinomial(probs, num_samples=1)
  else:
    new_token_ix = probs.argmax()

  return new_token_ix.item()



def generate_sentence(model, sentence, tokenizer, device, steps=50, temperature=1.0, sample=True, top_k=10, seed=None):
  if seed is not None:
    torch.manual_seed(seed)

  token_ixs = tokenizer.encode(sentence)
  token_ixs.insert(0, tokenizer.bos_token_id)

  for step in range(steps):
    new_token_ix = generate_token(model, token_ixs, device, temperature, sample, top_k)

    if new_token_ix == tokenizer.eos_token_id:
      break

    token_ixs.append(new_token_ix)

  text = tokenizer.decode(token_ixs[1:])
  text = ftfy.fix_text(text)
  print(text)



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



if __name__ == '__main__':
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

  print(count_parameters(model))

  tokenizer = load_tokenizer()

  prompt = input('Prompt: ')

  generate_sentence(model, prompt, tokenizer)
