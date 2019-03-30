import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine
import pickle
from utils import tokenize
import os
from collections import Counter
import itertools
import random

app = Flask(__name__)

# load data
db_path = os.path.abspath('../data/DisasterResponse.db')
engine = create_engine('sqlite:///' + db_path)

df = pd.read_sql_table('messages', engine)

# load model
with open("../models/classifier.pkl", 'rb') as f:
    model = pickle.load(f)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals    

    #Visual 1: Genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #Visual 2: Word cloud
    df = df.assign(cleaned_tokens = df['message'].apply(tokenize).apply(lambda x: list(filter(lambda x: not(x.isnumeric()), x)) ))
    all_messages_combined_words_cleaned = list( itertools.chain.from_iterable(df['cleaned_tokens']) )
    word_count_dict = dict( Counter(all_messages_combined_words_cleaned) )
    word_count_df = pd.DataFrame.from_dict(word_count_dict, orient = 'index', columns = ['count'])
    word_count_df['frequency'] = word_count_df['count'].apply(lambda x: x/len( all_messages_combined_words_cleaned ))
    word_count_df = word_count_df.sort_values('frequency', ascending=False)
    word_count_df = word_count_df[:100]
    min_freq = word_count_df.frequency.min()
    max_freq = word_count_df.frequency.max()
    word_count_df = word_count_df.assign( score = word_count_df.frequency.apply(
        lambda x: 15 + ( ((x-min_freq)/(max_freq - min_freq))*20 )
    ))
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(100)]   

    # create visuals    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data' : [
                Scatter(x=[random.random() for i in range(100)],
                    y=[random.random() for i in range(100)],
                    mode='text',
                    text=word_count_df.index,
                    marker={'opacity': 0.3},
                    textfont={'size': word_count_df.score,
                            'color': colors})
            ],
            'layout' : {
                'xaxis': {
                    'showgrid': False,
                    'showticklabels': False,
                    'zeroline': False
                },
                'yaxis': {
                    'showgrid': False, 
                    'showticklabels': False, 
                    'zeroline': False
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()