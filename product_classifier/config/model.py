class ConfigModel:
    image_width = 256
    image_height = 256

    resnet_output_units = 50

    word_embedding_file_path = ''
    word_embedding_vector_length = 50
    bad_words = []  # additional words to be removed in title embedding process e.g. ['small', 'medium', 'large']
    excluded_tokens = '-/.%'
