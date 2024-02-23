FROM python:3.8-slim

WORKDIR /usr/src/app

COPY model_evaluator.py .

RUN pip install --no-cache-dir numpy scipy pandas matplotlib seaborn scikit-learn transformers torch sentence-transformers youtube-transcript-api spacy nltk 

EXPOSE 80

ENV NAME model_evaluator
