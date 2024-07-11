def load_custom_pretrained_weights(custom_model, pretrained_model_name_or_path):
    from transformers import BartModel

    # Load the pretrained BART model
    pretrained_model = BartModel.from_pretrained(pretrained_model_name_or_path)

    # Manually copy the weights from the pretrained model to your custom model
    custom_model_dict = custom_model.state_dict()
    pretrained_dict = pretrained_model.state_dict()

    # Filter out unnecessary keys and update your custom model's state dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_model_dict}
    custom_model_dict.update(pretrained_dict)

    # Load the new state dict into your custom model
    custom_model.load_state_dict(custom_model_dict)

    return custom_model

# Example usage
custom_model = MyBartModel(config)
custom_model = load_custom_pretrained_weights(custom_model, 'bart-large')
