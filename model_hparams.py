class ModelHParams():
  def __init__(
    self,
    vocab_size=50257,
    n_positions=512,
    n_embd=768,
    n_layer=8,
    n_head=12,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-05,
    bos_token_id = 2,
    eos_token_id = 2
  ):

    self.vocab_size = vocab_size
    self.n_positions = n_positions
    self.n_embd = n_embd
    self.n_layer = n_layer
    self.n_head = n_head
    self.resid_pdrop = resid_pdrop
    self.embd_pdrop = embd_pdrop
    self.attn_pdrop = attn_pdrop
    self.layer_norm_epsilon = layer_norm_epsilon
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id