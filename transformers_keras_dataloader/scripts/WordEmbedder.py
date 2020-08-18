import os
import warnings
import json
import pathlib
import numpy as np
import torch

from .OffsetMapper import OffsetMapper
from .PoolPolicy import PoolPolicy


class WordEmbedder(object):
  def __init__(self, max_length, oov='avg', infer_oov_after_embed=False, all_special_tokens=None, pad_token_id=None,
               pooling_layer_number=-1, policy_dict=None,
               common_params=None, model_x_vars_params=None,
               is_tokenizer_fast=None, use_gpu=False, call_by_dataloader=False):
    r"""
    A simple WordEmbedder class which can be used to generate word embeddings for text sequences by utilizing huggingface's transformers package.

    Args:
        max_length (:obj:`Union[int, None]`, required):  
          Control the length for padding/truncation.
        oov (:obj:`str`, optional, defaults to :obj:`None`):
          Specify from predefined list of strategies on how to handle out of context words; only useful when generating word embeddings
          (Eg: Specify any one strategy from ['avg', 'sum', 'last'])
        infer_oov_after_embed (:obj:`bool`, optional, defaults to "False"):
          Infer out of vocab word lengths before passing through the pretrained model;
          If set to True, we perform extra padding above the max_length threshold since we have prior knowledge about full length of sequence with oov words
          We do this inorder to skip explicit padding after passing through the pretrained model.
        all_special_tokens (:obj:`list`, optional, defaults to :obj:`None`):  
          List of special tokens used by the model.
        pad_token_id (:obj:`int`, optional, defaults to :obj:`None`):  
          Token id maintained for the pad token by respective pretrained model.
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
    self.oov = oov                                          # str (Eg: any one from ['avg', 'sum', 'last'])
    self.infer_oov_after_embed = infer_oov_after_embed      # bool
    self.common_params = common_params                      # dict
    self.model_x_vars_params = model_x_vars_params          # dict
    self.all_special_tokens = all_special_tokens            # list
    self.pad_token_id = pad_token_id                        # int 
    self.is_tokenizer_fast = is_tokenizer_fast              # bool (defaults to None)
    self.use_gpu = use_gpu                                  # bool
    self.call_by_dataloader = call_by_dataloader            # bool
    self.pool_policy = PoolPolicy(pooling_layer_number, policy_dict)

    if self.all_special_tokens is None:
      self.all_special_tokens = []

    if self.oov is None:
      self.oov = 'avg'

    if (self.common_params is None) or (self.model_x_vars_params is None):
      self.offset_mapper = None
      self.encoder_params = None
      self._get_params()
    else:
      # Since this a word embedder we need an algorithm to map tokenized indices to reference tokens(text split using split())
      self.offset_mapper = OffsetMapper(offset_config = self.model_x_vars_params['offset_mapper_config'],
                                        all_special_tokens = self.all_special_tokens)
      # Prepare encoder params
      self.encoder_params = {**self.common_params['encoder_params'], **self.model_x_vars_params['parameter_overrides']['encoder_params']}

  def prepare_embeddings(self, X, tokenizer, model):
    if not self.call_by_dataloader:
      # Get all_special_tokens
      self.all_special_tokens = tokenizer.all_special_tokens
      # Get pad token id
      if (tokenizer.pad_token_id is None):
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
      self.pad_token_id = tokenizer.pad_token_id
      # Identify if tokenizer is fast
      self.is_tokenizer_fast = tokenizer.is_fast
      # Get model vars and params
      self.model_x_vars_params = self.model_vars_params_dict[model.config.model_type]
      # Since this a word embedder we need an algorithm to map tokenized indices to reference tokens(text split using split())
      self.offset_mapper = OffsetMapper(offset_config = self.model_x_vars_params['offset_mapper_config'], all_special_tokens = self.all_special_tokens)
      # Prepare encoder params
      self.encoder_params = {**self.common_params['encoder_params'], **self.model_x_vars_params['parameter_overrides']['encoder_params']}

    # Container to store word embedded matrix
    embedded_X = []

    # Iterate over all the sequences
    for sequence in X:
      # Encode each sequence 
      encoded_dict = tokenizer.encode_plus(text=sequence,
                                          max_length = self.max_length,  # Used to truncate or shorten the sequence
                                          **self.encoder_params)
      # Get token mapping
      token_mapping = None
      if self.is_tokenizer_fast:
        token_mapping = self.offset_mapper.map_offsets(sequence_str = sequence,
                                       position_list = encoded_dict['offset_mapping'])
      else:
        token_mapping = self.offset_mapper.map_offsets(sequence_str = sequence,
                                       tokenized_list = tokenizer.convert_ids_to_tokens(encoded_dict['input_ids']))

      # Infer out of vocab word lengths before passing through the pretrained model;
      # here we perform extra padding over the max_length threshold since we have prior knowledge of full length of sequence with oov
      # We do this inorder to skip explicit padding. 
      if not self.infer_oov_after_embed:
        # Compute extra padding length
        extra_padding_length = len(token_mapping) - len(set(token_mapping))
        # pad input_ids with pad token
        encoded_dict['input_ids'] = encoded_dict['input_ids'] + [self.pad_token_id] * ((self.max_length + extra_padding_length) - len(encoded_dict['input_ids']))
        # pad attention_mas with 0 
        encoded_dict['attention_mask'] = encoded_dict['attention_mask'] + [0] * ((self.max_length + extra_padding_length) - len(encoded_dict['attention_mask']))
        # Since we padded it we also modify token_mapping to reflect this change
        token_mapping = token_mapping + list(range(token_mapping[-1]+1, (token_mapping[-1]+1)+(len(encoded_dict['input_ids'])-len(token_mapping))))

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
      word_embed_matrix_tensor = self.pool_policy.pool(output_tuple[-1][1:])

      # [Discard]
      output_tuple = None

      # Considering only the relevant tokens with respect to word_embed_matrix_tensor
      token_mapping = token_mapping[:word_embed_matrix_tensor.shape[0]]

      # Perform oov pooling and append to embedded_batch list
      state_tensor = self._handle_oov(token_mapping, word_embed_matrix_tensor)
        
      # If opted for explicit padding we pad zero vectors to maintain shape
      if self.infer_oov_after_embed:
        # In order to have equal shape wrt other sequences in the batch or dataset we need to pad state_tensor with vectors filled with zeros(float)
        # We first construct a tensor frame
        output_tensor = torch.zeros((self.max_length, state_tensor.shape[1])) # For bert shape will be (max_length x 768)
        output_tensor[:state_tensor.shape[0], :state_tensor.shape[1]] = state_tensor
      else:
        output_tensor = state_tensor

      # Append embedded sequence to the container
      embedded_X.append(output_tensor.detach().numpy())

    return np.asarray(embedded_X)


  def _handle_oov(self, token_mapping, word_embed_matrix_tensor):
    # Construct a output tensor frame; shape[num_of_tokens, embedding_size]
    state_output_tensor = torch.zeros((np.max(token_mapping)+1, word_embed_matrix_tensor.shape[1]))

    # oov token split count array
    split_count_array = np.unique(token_mapping, return_counts=True)[1] # consider only the counts to get idea of splits

    # Convert token_mapping and split_count_array to tensor
    token_mapping = torch.as_tensor(token_mapping, dtype=torch.int64)
    split_count_array = torch.as_tensor(split_count_array, dtype=torch.float64)

    if self.use_gpu and torch.cuda.is_available():
      state_output_tensor = state_output_tensor.to(self.device)
      token_mapping = token_mapping.to(self.device)
      split_count_array = split_count_array.to(self.device)

    if self.oov == 'last':
      state_output_tensor += word_embed_matrix_tensor[np.cumsum(split_count_array)-1]
    else:
      # Sum vectors corresponding to oov token indices into respective token index (1D Eg: [1, 2, 2, 3, 3, 3, 4] will be [1, 2+2, 3+3+3, 4])
      state_output_tensor.index_add_(dim=0, index=token_mapping, source=word_embed_matrix_tensor)
      if self.oov == 'avg':
        # Simply average state_output_tensor using the split_count_array
        state_output_tensor /= split_count_array[:, None]

    # [Discard] some tensors
    token_mapping = None
    split_count_array = None
    word_embed_matrix_tensor = None

    return state_output_tensor

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