import pytest
from model import CatDogClassifier
import os
def test_model_loads():
    classifier = CatDogClassifier()
    assert classifier.model is not None
def test_prediction_format():
    classifier = CatDogClassifier()
    from PIL import Image
    img = Image.new('RGB', (224, 224), color='red')
    img.save('test_image.jpg')
    label, confidence = classifier.predict('test_image.jpg')
    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    os.remove('test_image.jpg')