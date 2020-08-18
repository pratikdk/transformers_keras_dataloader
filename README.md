# Transformers Keras Dataloader 🔌
**Transformers Keras Dataloader** provides an EmbeddingDataLoader class, a subclass of **keras.utils.Sequence** which enables real-time data feeding to your Keras model via ***batches***, hence making it possible to train with large datasets while overcoming the problem of loading the entire dataset in the memory prior to training.

**EmbeddingDataLoader** inherently is a generator which works by implementing functions required by Keras to get new batch of data from your dataset while fitting and predicting. We leverage this generator concept by real time processing of data while fitting and predicting, which unlocks the capacity to **handle bigger datasets** and use **larger batch size**.
When generating a training batch, for each sequence in the batch we get its embedding (either ***word embedding*** ***or sentence embedding***) by utilizing [**Huggingface's transformers**](https://huggingface.co/transformers/index.html) package.

We have also given option(parameter) to **utilize** **GPU**(if available) for both data(storing/processing) and model(forward pass), Aditionally you can also utilize **multiprocessing** to proces your dataset on multiple cores in real time and feed it right away to your downstream model while fitting.

This package also provides support for custom **'Pooling Strategies & Layer Choices'** hence enabling you to feed different vector combinations as input features to your downstream model. This is partially demonstrated through prior experiments, that different layers of pretrained model encode very different kinds of information, so the appropriate pooling layer or strategy will change depending on the application because different layers encode different kinds of information, please look at Arguments or usage section for info on how to utilize.

##  Installation
```sh
$ pip install transformers-keras-dataloader
```
##  Usage

### Using EmbeddingDataLoader class

Train by using EmbeddingDataLoader class to generate **word embeddings** for downstream training
```
from transformers_keras_dataloader import load_pretrained_model_and_tokenizer
from transformers_keras_dataloader import EmbeddingDataLoader


tokenizer, model = load_pretrained_model_and_tokenizer(pretrained_model_name_or_path = "bert-base-uncased",
                                                       return_model = True,
                                                       use_cuda = True)


train_dataloader = EmbeddingDataLoader(embedding_type="word", model=model, tokenizer=tokenizer,
                                       X=train_X, batch_size=64,
                                       max_length=100, sampler='random',
                                       y=train_y, num_classes=2, get_one_hot_label=True,
                                       use_gpu=True,
                                       pooling_layer_number=11, policy_dict=None, oov='avg', infer_oov_after_embed=False)

val_dataloader = EmbeddingDataLoader(embedding_type="word", model=model, tokenizer=tokenizer,
                                     X=val_X, batch_size=64,
                                     max_length=100, sampler='random',
                                     y=val_y, num_classes=2, get_one_hot_label=True,
                                     use_gpu=True,
                                     pooling_layer_number=11, policy_dict=None, oov='avg', infer_oov_after_embed=False)

test_dataloader = EmbeddingDataLoader(embedding_type="word", model=model, tokenizer=tokenizer,
                                      X=test_X, batch_size=64,
                                      max_length=100, sampler='random',
                                      use_gpu=True,
                                      pooling_layer_number=11, policy_dict=None, oov='avg', infer_oov_after_embed=False)

# Define Downstream model
def model_arch_for_word_embeddings(embedding_vector_len, num_output_neurons):
	model = Sequential()
	model.add(Dense(1500, activation='relu', batch_input_shape=(None, None, embedding_vector_len))) 
	[...] # Architecture
	model.add(Dense(num_output_neurons, activation='sigmoid'))
	model.compile()
	return model

downstream_model = model_arch_for_word_embeddings(embedding_vector_len=768,
												  num_output_neurons=2) 

## since bert outputs 768-hidden

# Fit 
history = downstream_model.fit(train_dataloader, val_dataloader=val_dataloader, epochs=2)
# Predict
predictions = downstream_model.predict(test_dataloader)
```

Train by using EmbeddingDataLoader class to generate **sentence embeddings** for downstream training
```
from transformers_keras_dataloader import load_pretrained_model_and_tokenizer
from transformers_keras_dataloader import EmbeddingDataLoader


tokenizer, model = load_pretrained_model_and_tokenizer(pretrained_model_name_or_path = "bert-base-uncased",
                                                       return_model = True,
                                                       use_cuda = True)


train_dataloader = EmbeddingDataLoader(embedding_type="sentence", model=model, tokenizer=tokenizer,
                                       X=train_X, batch_size=64,
                                       max_length=100, sampler='random',
                                       y=train_y, num_classes=2, get_one_hot_label=True,
                                       use_gpu=True,
                                       pooling_layer_number=11, policy_dict=None)

val_dataloader = EmbeddingDataLoader(embedding_type="sentence", model=model, tokenizer=tokenizer,
                                     X=val_X, batch_size=64,
                                     max_length=100, sampler='random',
                                     y=val_y, num_classes=2, get_one_hot_label=True,
                                     use_gpu=True,
                                     pooling_layer_number=11, policy_dict=None)

test_dataloader = EmbeddingDataLoader(embedding_type="sentence", model=model, tokenizer=tokenizer,
                                      X=test_X, batch_size=64,
                                      max_length=100, sampler='random',
                                      use_gpu=True,
                                      pooling_layer_number=11, policy_dict=None)

# Define Downstream model
def model_arch_for_sentence_embeddings(embedding_vector_len, num_output_neurons):
	model = Sequential()
	model.add(Dense(1500, activation='relu', batch_input_shape=(None, embedding_vector_len))) 
	[...] # Architecture
	model.add(Dense(num_output_neurons, activation='sigmoid'))
	model.compile()
	return model

downstream_model = model_arch_for_word_embeddings(embedding_vector_len=768,
                                                  num_output_neurons=2) 

## since bert outputs 768-hidden

# Fit 
history = downstream_model.fit(train_dataloader, val_dataloader=val_dataloader, epochs=2)
# Predict
predictions = downstream_model.predict(test_dataloader)
```
### Using WordEmbedder class

A simple **WordEmbedder** class which can be used to generate word embeddings for text sequences by utilizing **Huggingface's transformers** package.
```
from transformers_keras_dataloader import load_pretrained_model_and_tokenizer
from transformers_keras_dataloader import WordEmbedder


tokenizer, model = load_pretrained_model_and_tokenizer(pretrained_model_name_or_path = "bert-base-uncased",
                                                       return_model = True,
                                                       use_cuda = True)

word_embedder = WordEmbedder(max_length=100)

# Generate word embeddings for X 
word_embeddings = word_embedder.prepare_embeddings(X=X, tokenizer=tokenizer, model=model)
```
### Using SentenceEmbedder class

A simple **SentenceEmbedder** class which can be used to generate sentence embeddings for text sequences by utilizing **Huggingface's transformers** package.
```
from transformers_keras_dataloader import load_pretrained_model_and_tokenizer
from transformers_keras_dataloader import SentenceEmbedder


tokenizer, model = load_pretrained_model_and_tokenizer(pretrained_model_name_or_path = "bert-base-uncased",
                                                       return_model = True,
                                                       use_cuda = True)

sentence_embedder = SentenceEmbedder(max_length=100)

# Generate sentence embeddings for X 
sentence_embeddings = sentence_embedder.prepare_embeddings(X=X, tokenizer=tokenizer, model=model)
```
### Additional Features ⭐
#### Support for **Pooling Strategies & Layer Choices**
You can utilize either `pooling_layer_number` or `policy_dict`
*Examples:*
```
pooling_layer_number = -1
```
**or** 
```
# Full fledged control by defining custom pooling strategy
# if policy_dict is set to the dataloader we use it instead of pooling_layer_number
policy_dict = {
    'base_expression': "9 + 10 + 11 + 12", # Layer numbers
    'norm_op': ('avg', [0.25, 0.25, 0.25, 0.25]) # Weights
}
```
#### Support for Multiprocessing
While fitting you can also utitlize `use_multiprocessing=True` and specify number of `workers`
```
downstream_model.fit(train_dataloader,
				  	 val_dataloader=val_dataloader,
				  	 epochs=1,
				  	 workers=6,
				  	 use_multiprocessing=True)
```

## Supported pretrained models

You can refer to below links to know about supported **Huggingface's** pretrained models, they have collated the supported models and their description.

For full list of the currently provided pretrained models by Huggingface, refer to [https://huggingface.co/transformers/pretrained_models.html](https://huggingface.co/transformers/pretrained_models.html)

For full list of community-uploaded models, refer to [https://huggingface.co/models](https://huggingface.co/models).