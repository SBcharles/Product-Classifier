class ConfigModel:
    image_width = 256
    image_height = 256

    resnet_output_units = 50

    word_embedding_file_path = '/home/ubuntu/data_store/trained_models/nate.blackbox.product.classifier/pretrained_word_vectors/glove.6B.50d.txt'
    word_embedding_vector_length = 50
    bad_words = []  # additional words to be removed in title embedding process e.g. ['small', 'medium', 'large']
    excluded_tokens = '-/.%'
