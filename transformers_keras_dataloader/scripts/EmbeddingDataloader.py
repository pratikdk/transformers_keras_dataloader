import os
import random
import warnings
import json
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import torch
from keras.utils import to_categorical

from .WordEmbedder import WordEmbedder
from .SentenceEmbedder import SentenceEmbedder

class EmbeddingDataLoader(Sequence):
  def __init__(self, embedding_type, model, tokenizer, X, batch_size, max_length=100, sampler='sequential',
               y=None, num_classes=None, get_one_hot_label=False,
               use_gpu=False,
               pooling_layer_number=-1, policy_dict=None, oov=None, infer_oov_after_embed=False,
               random_seed=None):
    r"""
    EmbeddingDataLoader is a subclass of keras.utils.Sequence which enables real-time data feeding to your Keras model via batches,
    hence making it possible to train with large datasets while overcoming the problem of loading the entire dataset in the memory prior to training.
    EmbeddingDataLoader inherently is a generator which works by implementing functions required by Keras to get new batch of data from your dataset while fitting and predicting.
    We leverage this generator concept by real time processing of data while fitting and predicting, which unlocks the capacity to handle bigger datasets and use larger batch size.
    When generating a training batch, for each sequence in the batch we get its embedding (either word embedding or sentence embedding) by utilizing huggingface's transformers package.
    We have also given option(param) to utilize the GPU(if available) for both data(storing/processing) and model(forward pass), Aditionally you can also utilize multiprocessing
    to proces your dataset on multiple cores in real time and feed it right away to your downstream model while fitting.
    This package also provides support for custom 'Pooling Strategies & Layer Choice' hence enabling you to feed different vector combinations as input features to your downstream model,
    This is partially demonstrated through prior experiments, that different layers of pretrained model encode very different kinds of information, so the appropriate pooling layer or strategy 
    will change depending on the application because different layers encode different kinds of information, please look at Arguments or usage section for info on how to utilize.


    Args:
        embedding_type (:obj:`str`, required, pass either "sentence" or "word"):
          Type of embeddings you want to generate for the downstream model, the text sequence can be either processed into `word` embeddings or `sentence` embeddings.
        model (:obj:`transformers.PreTrainedModel`, required):
          Instantiated pretrained model object. Please first make use of `transformers_keras_dataloader.load_pretrained_model_and_tokenizer()` to instantiate both tokenizer and model objects.
        tokenizer (:obj:`transformers.PreTrainedTokenizer`, required):
          Instantiated pretrained model's tokenizer object. Please first make use of `transformers_keras_dataloader.load_pretrained_model_and_tokenizer()` to instantiate both tokenizer and model objects.
        X (:obj:`np.ndarray`, required):
          A Numpy array of input text sequences, which is basically your input
        batch_size (:obj:`int`, required):
          Batch size for your dataset. batch_size should be respective to your downstream model.
        max_length (:obj:`Union[int, None]`, optional, defaults to :obj:`None`):  
          Control the length for padding/truncation. 
        sampler (:obj:`int`, optional, defaults to "sequential"):
          Sequences for each batch are sampled based on type of sampler you set.
        y (:obj:`np.ndarray`, optional, defaults to :obj:`None`):
          A Numpy array of output labels, which is basically your output.
        num_classes (:obj:`int`, optional, defaults to :obj:`None`):
          Number of classes in your output. This is only required if you set `get_one_hot_label=True`, we use this param to construct a one hot vector matrix for labels.
        get_one_hot_label (:obj:`bool`, optional, defaults to "False"):
          If set to True, y for each batch is convert to a one hot vector matrix using keras's to_categorical().
        use_gpu (:obj:`bool`, optional, defaults to "False"):
          If set to True, it utilizes the GPU(if available) for both data(storing/processing) and model(forward pass).
        pooling_layer_number (:obj:`int`, optional, defaults to '-1'):
          Specify an integer (Eg: 11, 11 is the second last layer for a 12 layered bert_base); This also support negative indexing (Eg: -1, last layer of the model), Default=-1 (last hidden state)
        policy_dict (:obj:`dict`, optional, defaults to :obj:`None`):
          Specify a dictionary (Eg: {'base_expression': "10 + 11 + 12", 'norm_op': ('avg', [weights])}); Default = None;
          'norm_op' can be (avg|sum|last); when doing sum we don't normalize;
          for weighted average supply respective weights;
          Supply equal weights as [1, 1,...] when doing simple sum.
        oov (:obj:`str`, optional, defaults to :obj:`None`):
          Specify from predefined list of strategies on how to handle out of context words; only useful when generating word embeddings
          (Eg: Specify any one strategy from ['avg', 'sum', 'last'])
        infer_oov_after_embed (:obj:`bool`, optional, defaults to "False"):
          Infer out of vocab word lengths before passing through the pretrained model; Only useful when generating word embeddings
          If set to True, we perform extra padding above the max_length threshold since we have prior knowledge about full length of sequence with oov words
          We do this inorder to skip explicit padding after passing through the pretrained model.
        random_seed (:obj:`int`, optional, defaults to :obj:`None`):
          Seed to use for random number generation; affects the random sampler since we shuffle when random sampling, therefore It is recommended to keep in None.

    """
    # Validate attributes
    assert (isinstance(embedding_type, str)) and (embedding_type in ['sentence', 'word'])
    assert (isinstance(X, np.ndarray)) and (len(X) > 2)
    assert (isinstance(batch_size, int)) and (batch_size > 0)
    assert (max_length is None) or ((isinstance(max_length, int)) and (max_length > 1))
    assert (isinstance(sampler, str)) and (sampler in ['sequential', 'random'])
    assert (y is None) or (isinstance(y, np.ndarray))
    assert (num_classes is None) or ((isinstance(num_classes, int)) and (num_classes > 1))
    assert (isinstance(get_one_hot_label, bool))
    assert (isinstance(use_gpu, bool))
    assert (isinstance(pooling_layer_number, int))
    assert (policy_dict is None) or (isinstance(policy_dict, dict)) and (all(p_op in policy_dict.keys() for p_op in ['base_expression', 'norm_op']))
    assert (oov is None) or (isinstance(oov, str)) and (oov in ['avg', 'sum', 'last'])
    assert (isinstance(infer_oov_after_embed, bool))
    assert (random_seed is None) or (isinstance(random_seed, int))

    # Parameter value warning
    if (y is None) and ((num_classes is not None) or (get_one_hot_label == True)):
      warnings.warn("Ignoring values set for num_classes or get_one_hot_label, since y=None.")
    if (use_gpu) and (not torch.cuda.is_available()):
      warnings.warn("Ignoring value set for use_gpu=True, since GPU isn't available.")
      use_gpu = False
    if (embedding_type == 'word') and (oov is None):
      warnings.warn("embedding_type='word', but oov was passed as 'None'; continuing execution with oov='avg'")
      oov='avg'
    if (embedding_type == 'sentence') and ((oov is not None) or (infer_oov_after_embed == True)):
      warnings.warn("Ignoring values set for oov and infer_oov_after_embed, since embedding_type='sentence', these parameter won't be used.")
    if (y is not None):
      unique_classes = len(np.unique(y))
      if (num_classes is None) or (num_classes != unique_classes):
        warnings.warn(f"Incorrect value set for num_classes parameter; continuing execution with num_classes={unique_classes}")
        num_classes = unique_classes
      if (get_one_hot_label == False) and (unique_classes > 2):
        warnings.warn(f"Number of unique labels in y is {unique_classes}, but get_one_hot_label was passed as False; continuing execution with get_one_hot_label=True.")
        get_one_hot_label=True
    
    self.embedding_type = embedding_type                                # str (Eg: any one from ['sentence', 'word']) 
    self.X = X                                                          # nd.array
    self.batch_size = batch_size                                        # int
    self.sampler = sampler                                              # str (Eg: any one from ['random', 'sequential'])    
    self.y = y                                                          # nd.array
    self.num_classes = num_classes                                      # int
    self.get_one_hot_label = get_one_hot_label                          # bool (Default is False)
    self.use_gpu = use_gpu                                              # bool (Default is False)
    self.random_seed = random_seed                                      # int

    # Load configurations
    self.common_params_config_path = pathlib.Path(os.path.abspath(__file__)).parent.parent/'config/common_params.json'
    self.model_vars_params_config_path = pathlib.Path(os.path.abspath(__file__)).parent.parent/'config/model_vars_params.json'
    self.common_params_dict = None
    self.model_vars_params_dict = None
    ## Load json; let open() handle file read errors
    # common params
    with open(self.common_params_config_path) as f:
      self.common_params_dict = json.load(f)
    # model_vars_params
    with open(self.model_vars_params_config_path) as f:
      self.model_vars_params_dict = json.load(f)
      
    # Download model and tokenizer
    self.tokenizer = tokenizer
    self.model = model

    # validate max_length
    if (max_length is None) and (self.tokenizer.max_len > 512):
      max_length = self.tokenizer.max_len
      warnings.warn(f"max_length=None, continuing execution with setting max_length={max_length}. It is advisable to provide a sensible max_length inorder to generate an embedding matrix with has a shape thats easy for computation.")
    elif (max_length is None) and (self.tokenizer.max_len <= 512):
      max_length = self.tokenizer.max_len
      warnings.warn(f"max_length=None, continuing execution with setting max_length={max_length}.")
    if (max_length > self.tokenizer.max_len):
      warnings.warn(f"max_length={max_length} cannot be greater than the maximum length the pretrained model can support, continuing execution with explicitly setting max_length={self.tokenizer.max_len}.")
      max_length = self.tokenizer.max_len

    # Validate model base type is one of the supported base types
    self.model_base_type = self.model.config.model_type
    if self.model_base_type not in self.model_vars_params_dict.keys():
      raise ValueError(f'Value for model and tokenizer is invalid, please verify if this model is supported.')
    # If model is valid, select the relevant config from
    self.model_x_vars_params = self.model_vars_params_dict[self.model_base_type]

    # Validate embedding type support for the passed base model
    if (self.embedding_type == "word" and not self.model_x_vars_params['supports_word_embedding']) or (
        self.embedding_type == "sentence" and not self.model_x_vars_params['supports_sentence_embedding']):
      raise ValueError(f'model base type: {self.model_base_type} does not support {self.embedding_type} embeddings.')

    # If pad token is absent add it explicitly (eg: '[PAD]')
    if (self.embedding_type == 'word') and (infer_oov_after_embed == False) and (self.tokenizer.pad_token_id is None):
      self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Initialize the requested embedder
    self.embedder = None
    if embedding_type == "word":
      self.embedder = WordEmbedder(max_length = max_length, oov = oov, infer_oov_after_embed = infer_oov_after_embed,
                                   common_params = self.common_params_dict,
                                   model_x_vars_params = self.model_vars_params_dict[self.model_base_type],
                                   all_special_tokens = self.tokenizer.all_special_tokens,
                                   pad_token_id = self.tokenizer.pad_token_id,
                                   pooling_layer_number = pooling_layer_number,
                                   policy_dict = policy_dict,
                                   is_tokenizer_fast = self.tokenizer.is_fast,
                                   use_gpu = self.use_gpu,
                                   call_by_dataloader = True)
    else:
      self.embedder = SentenceEmbedder(max_length = max_length,
                                       common_params = self.common_params_dict,
                                       model_x_vars_params = self.model_vars_params_dict[self.model_base_type],
                                       pooling_layer_number = pooling_layer_number,
                                       policy_dict = policy_dict,
                                       is_tokenizer_fast = self.tokenizer.is_fast,
                                       use_gpu = self.use_gpu,
                                       call_by_dataloader = True)

    # Organize data based on sampler
    self._organize_data()
    # Simple iterator for outputting dummy batches to test how input X and y is structured before encoding or embedding.
    self.dummy_counter = -1
    print(f"Is the model fast: {self.tokenizer.is_fast}")


  def __len__(self):
    """
    (Eg: num_data_samples/batch_size = 230/16 = floor(14.375) = 14)
    :return: Return total number of batches in each Epoch
    """
    return int(np.floor(self.X.shape[0] / self.batch_size))


  def __getitem__(self, index):
    """
    Generates a batch

    :param index: index of batch
    :return: (X, y) batch when training and y when testing
    """
    X_batch = self.X[index*self.batch_size : (index+1)*self.batch_size]

    # Encode and Embed the batch(each sequence in the batch)
    embedded_X_batch_array = self.embedder.prepare_embeddings(X_batch, self.tokenizer, self.model)

    # [Discard] some tensors
    X_batch = None 

    # If y is passed, we batch it and return the (X, y) batch 
    if self.y is not None:
      # Also batch up Y
      y_batch = self.y[index*self.batch_size : (index+1)*self.batch_size]
      # Also generate one hot matrix for Y, if requested # num_classes required to be set
      if self.get_one_hot_label:
        y_batch = to_categorical(y=y_batch, num_classes=self.num_classes)
      else:
        y_batch = y_batch
      # finally return batch of X along with y
      return embedded_X_batch_array, y_batch
    else:
      return embedded_X_batch_array


  def on_epoch_end(self):
    # RE-Organize data, only really useful when sampler is 'random' and random seed is 'None'
    self._organize_data()


  def _organize_data(self):
    if self.sampler == "random":
      np.random.seed(self.random_seed)
      order = np.random.permutation(len(self.X))
      self.X = self.X[order]
      self.y = self.y[order]


  def get_dummy_batch_for_testing(self):
    # Returns the  batch (examples selected sequentially)
    self.dummy_counter += 1
    print(f'Batch No: {self.dummy_counter}')
    return self.__getitem__(self.dummy_counter)


  def get_total_batch_count(self):
    print(f'Number of batches: {self.__len__()}')
