dataset = "imdb"  # @param ["yelp", "imdb"]

term_embd_category = "from_scratch"  # @param ["from_scratch", "pre_trained"]
term_vect_mdl = "fasttext"  # @param ["fasttext"]
recurrent_neural_network_type = "GRU"  # @param ["LSTM", "GRU"]
lr = 0.001
epochs = 8
batch_size = 64