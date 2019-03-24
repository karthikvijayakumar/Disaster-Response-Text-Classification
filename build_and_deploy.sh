python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
python train_classifier.py data/DisasterResponse.db models/classifier.pkl
gunicorn --chdir app run:app
