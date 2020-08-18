import numpy as np
import torch

class PoolPolicy(object):
  def __init__(self, pooling_layer_number, policy_dict):
    self.pooling_layer_number = pooling_layer_number    # int (Eg: 11, 11 is the second last layer for a 12 layered bert_base, range of values [1, max(layer_number)]) ; Default=-1 (last hidden state)
    self.policy_dict = policy_dict                      # dict (Eg: {'base_expression': "10 + 11 + 12", 'norm_op': ('avg', [weights])}); Default = None;
                                                        # 'norm_op' can be (avg|sum|last); when doing sum we don't normalize;
                                                        # for weighted average supply respective weights;
                                                        # Supply equal weights as [1, 1,...] when doing simple sum
    self.supported_ops = {                              # Predefined operations that base expression can support
      "+": torch.add,
      "-": torch.sub,
      "/": torch.true_divide,
      "*": torch.mul
    }
    self.policy_evaluator = 'iterative'                 # str (Eg: any one from ['iterative', 'recursive'])

  def pool(self, hidden_states_tuple):
    word_embed_matrix_tensor = None

    if not self.policy_dict is None: # if policy dict was passed we use it instead of pooling_layer_number
      # Extract and split base expression to make it parseable
      base_expression = self.policy_dict['base_expression'].strip().split()
      # Extract custom policy dictionary, contains avg
      norm_op = self.policy_dict['norm_op']

      # First evaluate base expression with respective weights
      if self.policy_evaluator == 'iterative':
        word_embed_matrix_tensor = self.iterative_eval_base_op(mat_tuple=hidden_states_tuple,
                                                              expression=base_expression,
                                                              weights=norm_op[1])
      else: # Recursive base expression evaluator
        word_embed_matrix_tensor = self.recursive_eval_base_op(mat_tuple=hidden_states_tuple,
                                                              expression=base_expression[::-1], # Recursive works inside out(R->L), but we want left to right; so we flip the list
                                                              weights=norm_op[1][::-1]) # Recursive works inside out(R->L), but we want left to right; so we flip the list
      # Normalize matrix when 'avg'(averaging) with the supplied weights
      if norm_op[0] == 'avg': # not needed while summing
        # Average the word_embed_matrix_tensor
        word_embed_matrix_tensor = word_embed_matrix_tensor / sum(norm_op[1])

    else:
      # if pooling_layer_number < 0 it means we want to select final layer states
      if ((self.pooling_layer_number > 0) and (self.pooling_layer_number > len(hidden_states_tuple))) or (
          (self.pooling_layer_number < 0) and (-self.pooling_layer_number > len(hidden_states_tuple))):
        raise ValueError(f'Value for pooling_layer_number={self.pooling_layer_number} is invalid, since available layers are {len(hidden_states_tuple)}.')
      # Making values >0 compatible with zero indexing
      if self.pooling_layer_number > 0:
        self.pooling_layer_number -= 1 
      # Choose hidden state layer specified by 'pooling_layer_number'
      # Also discard state vectors for [CLS] and [SEP] tokens
      word_embed_matrix_tensor = hidden_states_tuple[self.pooling_layer_number][0]
    
    # Return pooled word_embed_matrix_tensor
    return word_embed_matrix_tensor.squeeze(0)



  # Iterative base policy evaluator
  def iterative_eval_base_op(self, mat_tuple, expression, weights):
    eval_value = None
    weight_ctr = 0
    for index in range(1, len(expression)-1):
      if eval_value is None:
        eval_value = mat_tuple[int(expression[0])-1] * weights[weight_ctr]
      expression_element = expression[index]
      if not expression_element.isdigit():
        if expression_element in self.supported_ops.keys():
          weight_ctr += 1
          eval_value = self.supported_ops[expression_element](eval_value,
                                                              mat_tuple[int(expression[index+1])-1] * weights[weight_ctr])
    return eval_value
  
  # Recursive base policy evaluator
  def recursive_eval_base_op(self, mat_tuple, expression, weights, weight_ctr=0):
    if len(expression) == 1:
      return mat_tuple[int(expression[0])-1] * weights[weight_ctr]
    
    return self.supported_ops.get(expression[1])(mat_tuple[int(expression[0])-1] * weights[weight_ctr],
                                                recursive_eval_base_op(mat_tuple, expression[2:], weights=weights, weight_ctr=weight_ctr+1))
    