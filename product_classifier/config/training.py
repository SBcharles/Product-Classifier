class ConfigTraining:

    dataset_dir = '/home/ubuntu/data_store/training_data/nate.blackbox.product.classifier.model/amazon_dataset'
    model_weights_dir = '/home/ubuntu/data_store/trained_models/nate.blackbox.product.classifier/classifier_models'

    max_products = 1000

    classes_to_keep = [
        'Books',
        'Clothing, Shoes & Jewelry',
        'Sports & Outdoors',
        'Electronics',
        'Home & Kitchen',
        'Automotive',
        'Cell Phones & Accessories',
        'Toys & Games',
        'Tools & Home Improvement',
        'CDs & Vinyl',
        # 'Beauty',
        # 'Health & Personal Care',
        # 'Grocery & Gourmet Food',
        # 'Patio, Lawn & Garden'
    ]

    train_val_test_proportions = (0.7, 0.15, 0.15)
    assert sum(train_val_test_proportions) == 1.0

    batch_size = 32
    epochs = 2
    patience = 6
