import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing'
]

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
# tokenizer = Tokenizer(num_words = 100)

# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)

# Get the indices and print it
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences((sentences))
padded = pad_sequences(sequences, padding = "post")
test_data = [
    'I really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("Word_index: ", word_index)
print("Sequences: ", sequences)
print("padded_sequences: ", padded)
#print("Test sequences with new words: ", test_seq)