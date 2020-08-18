import os
import warnings
import json
import pathlib
import numpy as np
import torch

from .PoolPolicy import PoolPolicy

class SentenceEmbedder(object):
  def __init__(self, max_length,
              pooling_layer_number=-1, policy_dict=None,
              common_params=None, model_x_vars_params=None,
              is_tokenizer_fast=False, use_gpu=False, call_by_dataloader=False):
    r"""
    A simple SentenceEmbedder class which can be used to generate sentence embeddings for text sequences by utilizing huggingface's transformers package.

    Args:
        max_length (:obj:`Union[int, None]`, required):  
          Control the length for padding/truncation.
        pooling_layer_number (:obj:`int`, optional, defaults to '-1'):
          Specify an integer (Eg: 11, 11 is the second last layer for a 12 layered bert_base); This also support negative indexing (Eg: -1, last layer of the model), Default=-1 (last hidden state)
        policy_dict (:obj:`dict`, optional, defaults to :obj:`None`):
          Specify a dictionary (Eg: {'base_expression': "10 + 11 + 12", 'norm_op': ('avg', [weights])}); Default = None;
          'norm_op' can be (avg|sum|last); when doing sum we don't normalize;
          for weighted average supply respective weights;
          Supply equal weights as [1, 1,...] when doing simple sum.
        common_params (:obj:`dict`, optional, defaults to :obj:`None`):  
          Generic parameters utilized during encoding using the pretrained model.
        model_x_vars_params (:obj:`dict`, optional, defaults to :obj:`None`):  
          Model specific variables and parameters, utilized during encoding and embedding using the pretrained model.
        is_tokenizer_fast (:obj:`bool`, optional, defaults to :obj:`None`):  
          Specify if the tokenizer object instantiated is an instance of Fast version of transformers tokenizer.
          Please first make use of `transformers_keras_dataloader.load_pretrained_model_and_tokenizer()` to instantiate both tokenizer and model objects.
        use_gpu (:obj:`bool`, optional, defaults to "False"):
          If set to True, it utilizes the GPU(if available) for both data(storing/processing) and model(forward pass).
        call_by_dataloader (:obj:`bool`, optional, defaults to :obj:`False`):  
          This parameter is used by EmbeddingDataLoader and need not be manually set.

    """
    self.max_length = max_length                            # int
    self.common_params = common_params                      # dict
    self.model_x_vars_params = model_x_vars_params          # dict
    self.is_tokenizer_fast = is_tokenizer_fast              # bool
    self.use_gpu = use_gpu                                  # bool
    self.call_by_dataloader = call_by_dataloader            # bool
    self.pool_policy = PoolPolicy(pooling_layer_number, policy_dict)

    if (self.common_params is None) or (self.model_x_vars_params is None):
      self.encoder_params = None
      self._get_params()
    else:
      # Prepare encoder params
      self.encoder_params = {**self.common_params['encoder_params'], **self.model_x_vars_params['parameter_overrides']['encoder_params']}
      # Modify some paramters from encoder_params
      self.encoder_params['pad_to_max_length'] = True

  def prepare_embeddings(self, X, tokenizer, model):
    if not self.call_by_dataloader:
      # Identify if tokenizer is fast
      self.is_tokenizer_fast = tokenizer.is_fast
      # Get model vars and params
      self.model_x_vars_params = self.model_vars_params_dict[model.config.model_type]
      # Prepare encoder params
      self.encoder_params = {**self.common_params['encoder_params'], **self.model_x_vars_params['parameter_overrides']['encoder_params']}
      # Modify some paramters from encoder_params
      self.encoder_params['pad_to_max_length'] = True

    # Container to store sentence embedding vectors for X
    embedded_X = []
        
    # Iterate over all the sequences
    for sequence in X:
      # Encode each sequence 
      encoded_dict = tokenizer.encode_plus(text=sequence,
                                          max_length = self.max_length,  # Used to truncate or shorten the sequence
                                          **self.encoder_params)
      # Wrap input_ids and attention_masks in tensor; to make them compatible for further operations
      input_ids = torch.tensor(encoded_dict['input_ids'])
      attention_mask = torch.tensor(encoded_dict['attention_mask'])

      # [Discard]
      encoded_dict = None

      # Place input_ids and attention_mask tensors on gpu if its specified and GPU is available
      if self.use_gpu and torch.cuda.is_available():
        # get GPU device
        self.device = torch.device('cuda')
        # Offload input_ids to gpu
        input_ids = input_ids.to(self.device)
        # Offload attention mask to gpu
        attention_mask = attention_mask.to(self.device)

      # Pass encoded sequence through the pretrained model;
      # Also handle inputs for text-to-text models
      if self.model_x_vars_params['requires_target_sequence']:
        output_tuple = model(input_ids = input_ids.unsqueeze(0),
                             attention_mask = attention_mask.unsqueeze(0),
                             decoder_input_ids = input_ids.unsqueeze(0))
      elif self.model_x_vars_params['requires_attention_mask']:
        output_tuple = model(input_ids = input_ids.unsqueeze(0),
                             attention_mask = attention_mask.unsqueeze(0))
      else:
        output_tuple = model(input_ids = input_ids.unsqueeze(0))
      
      # Apply custom pool policy on hidden states, We only pass hidden states, i.e output_tuple[-1][1:] skips output vector
      token_embed_matrix_tensor = self.pool_policy.pool(output_tuple[-1][1:])

      # [Discard]
      output_tuple = None

      # Calculate the average of all token vectors in token_embed_matrix_tensor.
      output_tensor = torch.mean(token_embed_matrix_tensor, dim=0)

      # Append embedded sequence to the container
      embedded_X.append(output_tensor.detach().numpy())

    return np.asarray(embedded_X)

  def _get_params(self):
    # Load configurations
    common_params_config_path = pathlib.Path(os.path.abspath(__file__)).parent.parent/'config/common_params.json'
    model_vars_params_config_path = pathlib.Path(os.path.abspath(__file__)).parent.parent/'config/model_vars_params.json'
    self.common_params_dict = None
    self.model_vars_params_dict = None
    ## Load json; let open() handle file read errors
    # common params
    with open(common_params_config_path) as f:
      self.common_params = json.load(f)
    # model_vars_params
    with open(model_vars_params_config_path) as f:
      self.model_vars_params_dict = json.load(f)