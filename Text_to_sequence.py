import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing'
    ]

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
#tokenizer = Tokenizer(num_words = 100)

# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)

# Get the indices and print it
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences((sentences))

test_data = [
    'I really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("Word_index: ", word_index)
print("Sequences: ", sequences)
print("Test sequences with new words: ", test_seq)


""" 
The output of the above code is:
Word_index:  {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}
Sequences:  [[4, 2, 1, 3], [4, 2, 1, 6], [5, 2, 1, 3], [7, 5, 8, 1, 3, 9, 10]]
Test sequences with new words:  [[4, 2, 1, 3], [1, 3, 1]]
  
Word_index have details of all the encoding happened for each unique word from the sentences
Sequences contain the list in which each element is represented using its tokens 
For exmaple: "I love my dog" is represented as [4, 2, 1, 3] where each elemt is token for each word.

If you look a test_seq output the word 'I really love my dog' is represented by [4, 2, 1, 3] but,
its wrong because it has missed encoding word "really" hat because we have not tokenized it.

This issuecan be taken care by adding new argument "oov_token" in tokenizer function which can 
take care of new keywords encountered in test data and assign it with desired keyword 
in this code we have used "<oov>" as keyword to represent new words. 

  """
