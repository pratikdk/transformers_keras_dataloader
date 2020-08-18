import numpy as np


class OffsetMapper(object):
  def __init__(self, offset_config, all_special_tokens):
    # Unpack and set values based on the passed config
    self.offset_resolver_type = offset_config['offset_resolver']
    self.spacing_indicator_symbol = offset_config['spacing_indicator_symbol']
    self.subword_indicator_symbol = offset_config['subword_indicator_symbol']
    self.indicator_position = offset_config['indicator_position']
    self.bos_token = offset_config['bos_token'] 
    self.eos_token = offset_config['eos_token']
    self.all_special_tokens = all_special_tokens
    self.resolver_dict = {
        "position_offset_mapper": self.map_position_offsets,
        "sp_offset_mapper": self.map_sp_offsets,
        "bpe_offset_mapper": self.map_bpe_offsets,
    }
    self.available_indicator = self.spacing_indicator_symbol if self.spacing_indicator_symbol else self.subword_indicator_symbol
    

  def map_offsets(self, sequence_str, tokenized_list=None, position_list=None):
    # Split sequence into simplest format, which is using spaces
    # This list will be used as reference list to allocate indexes from
    ref_sequence_list = sequence_str.strip().split()
    x_list = tokenized_list
    # If position_list is passed used it instead of tokenized_list
    if position_list is not None:
      x_list = np.asarray(position_list)
    # Get offset mapping
    offsets_mapped_list = self.resolver_dict[self.offset_resolver_type](ref_sequence_list, x_list)
    return offsets_mapped_list


  def map_position_offsets(self, ref_list, position_list):
    # Subtract susequent subword end, start index to highlight spacing
    output_positions = np.subtract(position_list[1:, 0], position_list[:-1, 1])
    # Mask and supply special tokens value as 1; since they are complete words and have space before them
    output_positions[position_list[1:, 0] <= 0] = 1
    # Prepend a zero to the list to indicate a starting word and perform a cumulative sum designate indices compute on spacing
    output_positions = np.cumsum(np.pad(output_positions, (1, 0), 'constant'))
    # return the positions
    return output_positions.tolist()


  def map_sp_offsets(self, ref_list, tokenized_list):
    # Container to store the word refrence index for each token
    token_map = []
    spaced_token_counter = 0 # We maintain a spaced_token_counter instead of doing a cumulative sum; its easy this way and involves less conversions
    bos_was_first_token = False
    eos_was_last_token = False
    for token in tokenized_list: # Scan each token
      # Identify if token is a (eos or bos token) or if token consists space indicator symbol, if so increment the spaced_token_counter and append to token_map
      if (len(token) >= len(self.spacing_indicator_symbol)) and (((self.bos_token and token == self.bos_token) or (self.eos_token and token == self.eos_token)) or (
          (token[:len(self.spacing_indicator_symbol)] if (self.indicator_position == 'left') else token[-len(self.spacing_indicator_symbol):]) == self.spacing_indicator_symbol)):
        if not token_map:
          if (self.bos_token and token == self.bos_token):
            bos_was_first_token = True
          token_map.append(spaced_token_counter)
        else:
          if (self.eos_token and token == self.eos_token):
            eos_was_last_token = True
          spaced_token_counter += 1
          token_map.append(spaced_token_counter)
      # Token is not eos, bos token and also doesnt consist spacing indictor, so don't increment the spaced_token_counter and append last value of counter to token_map
      else:
        if bos_was_first_token or eos_was_last_token:
          spaced_token_counter += 1
          bos_was_first_token = False
          eos_was_last_token = False
        token_map.append(spaced_token_counter)
    return token_map

  def map_bpe_offsets(self, ref_list, tokenized_list):
    # print(f'Using indicator: {self.available_indicator}')
    token_map = []
    counter = 0 # We maintain a spacing based token counter instead of doing a cumulative sum; its easy this way and involves less conversions
    ref_scan_index = 0 # Starting index of the token map and also the first element from ref_list we will look at while parsing tokenized_list
    ref_scan_index_word_len = len(ref_list[ref_scan_index]) # length of the word at ref_scan_index in ref_list
    bos_was_first_token = False

    for token in tokenized_list: # Scan each token
      if (self.bos_token and token == self.bos_token) or (self.eos_token and token == self.eos_token): # Deal with bos and eos token
        if not token_map:
          token_map.append(counter)
          if (self.bos_token and token == self.bos_token):
            bos_was_first_token = True
        else:
          counter += 1
          token_map.append(counter)
      else: # Deal with all other type of tokens
        if bos_was_first_token:
          counter += 1
          bos_was_first_token = False
        # Extract relevant characters
        chars = ""
        if (len(token) >= len(self.available_indicator)) and ((token[:len(self.available_indicator)] if (self.indicator_position == 'left') else token[-len(self.available_indicator):]) == self.available_indicator):
          chars = (token[len(self.available_indicator):] if (self.indicator_position == 'left') else token[:-len(self.available_indicator)])
        else:
          chars = token

        # Identify characters
        if chars in self.all_special_tokens: # If its a special token (usually a single element)
          if ref_scan_index_word_len > 0: # If its part of the same chunk
            token_map.append(counter)
            ref_scan_index_word_len -= 1
          else: # so its not part of the same chunk; increment and append the counter
            counter += 1
            token_map.append(counter)
            buffer_subtraction_len = 1
            ## Switch to next word and subtract len of chars from it
            ref_scan_index += 1
            ref_scan_index_word_len = len(ref_list[ref_scan_index])
            ref_scan_index_word_len -= 1
        else: # its not part of the special characters
          if ref_scan_index_word_len > 0:
            token_map.append(counter)
            ref_scan_index_word_len -= len(chars)
          else:
            counter += 1
            token_map.append(counter)
            buffer_subtraction_len = len(chars)
            ## Switch to next word and subtract len of chars from it
            ref_scan_index += 1
            ref_scan_index_word_len = len(ref_list[ref_scan_index])
            ref_scan_index_word_len -= len(chars)
    # Return mapped indicies
    return token_map 