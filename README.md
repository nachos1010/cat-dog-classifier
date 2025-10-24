# Cat vs Dog Classifier
A web application that classifies images as cats or dogs using a pretrained MobileNetV2 m


## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Docker
```bash
docker build -t cat-dog-classifier .
docker run -p 8501:8501 cat-dog-classifier
```

## Testing
```bash
pytest tests/
```
