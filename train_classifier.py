import sys
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle
from utils import tokenize

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('Select * from messages;', con=engine)
    X = df['message']
    Y = df[list( df.columns )[4:]]
    category_names = list( df.columns )[4:]
    return [X,Y,category_names]

def build_model():
    pipeline = Pipeline([
        ('countvectorizer', CountVectorizer(tokenizer= tokenize)),
        ('tfidftransformer', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = 1))    
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_test_preds = model.predict(X_test)
    y_test_preds = pd.DataFrame(y_test_preds, columns= Y_test.columns, index=  Y_test.index)
    clf_rpt_test = pd.DataFrame(columns = ['f1-score', 'precision','recall', 'support', 'column'])
    for col in Y_test.columns:
        if(col == 'related'):
            col_report = classification_report( Y_test[col].values, y_test_preds[col].values, output_dict=True, target_names = ['False','True', 'Something'])
        elif(col == 'child_alone'):
            col_report = classification_report( Y_test[col].values, y_test_preds[col].values, output_dict=True, target_names = ['False'])
        else:
            col_report = classification_report( Y_test[col].values, y_test_preds[col].values, output_dict=True, target_names = ['False','True'])        
        df_temp = pd.DataFrame(col_report).transpose()
        df_temp['column'] = col       
        clf_rpt_test = pd.concat( [clf_rpt_test, df_temp] )
        
    clf_rpt_test = clf_rpt_test.reset_index().rename(index = str, columns = {'index': 'metric'} ).set_index(['column', 'metric']).sort_index()
    pd.options.display.max_rows = 999
    print(  clf_rpt_test.xs('micro avg', level = 1).sort_values('f1-score') )
    pd.reset_option( 'display.max_rows' )

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()