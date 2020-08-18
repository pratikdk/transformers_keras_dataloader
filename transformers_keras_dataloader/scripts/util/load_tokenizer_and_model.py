import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

def load_pretrained_model_and_tokenizer(pretrained_model_name_or_path, return_model=True, use_cuda=False):
  r"""
  A simple function to automatically retrieve the relevant model and tokenizer given the name/path to the pretrained weights/config/vocabulary
  This function uses HuggingFace's AutoModel, AutoConfig and AutoTokenizer classes from their transformers package.
  For more info on Auto classes: https://huggingface.co/transformers/model_doc/auto.html
  
  Args:
    pretrained_model_name_or_path: either:
      - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
      - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
      - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
    return_model (:obj:`bool`, optional, defaults to "True"):  
      Specify if model is also required, This should True if you also this function to return Pretrained Model along with Pretrained Tokenizer
    use_cuda (:obj:`bool`, optional, defaults to "False"):  
      Specify if you want to load Pretrained model on your GPU(it is faster this way); It is advisable to verify if GPU is available on the sytem being used.
      Although we do verify it and we only try use it if GPU device is available.
    
  """

  # Download tokenizer 
  tokenizer = AutoTokenizer.from_pretrained(
      pretrained_model_name_or_path = pretrained_model_name_or_path,
      use_fast=True)
  # Download Model
  model = None
  if return_model:
    # Download model config
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path = pretrained_model_name_or_path,
        output_hidden_states = True)

    # Download model using model config
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path = pretrained_model_name_or_path,
        config = model_config
    )
    # Utilize GPU if specified and available
    if use_cuda and torch.cuda.is_available():
      # Specify and select device as GPU
      device = torch.device('cuda')
      # Transfer model execution to GPU device
      model.to(device)

  return tokenizer, model