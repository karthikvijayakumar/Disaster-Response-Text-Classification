import sys
import pandas as pd
from sqlalchemy import create_engine
from collections import Counter
import itertools
from utils import tokenize

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories data from the given file paths, process them and merge them
    
    Args:
    messages_filepath: 
    categories_filepath:

    Returns:
    df: pandas.DataFrame: dataframe containing the messages data combined with their category classifications    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categories_split = categories.categories.str.split(';', expand=True)    

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(  categories_split.iloc[0].apply(lambda x: str.split(x,'-')[0]) )

    categories_split.columns = category_colnames

    categories = pd.concat( [ categories, categories_split ] , axis = 1)

    # Convert categories indicator columns to numeric
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str.split(x,'-')[1])
        
        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    
    df = pd.merge(messages, categories, on = 'id')    
    # Drop the categories column

    df = df.drop('categories', axis = 1)

    #Compute the word counts and frequencies for the top 100 words
    all_messages_combined_words_cleaned = list( 
        itertools.chain.from_iterable(
            df['message'].apply(tokenize).apply(lambda x: list(filter(lambda x: not(x.isnumeric()), x)) )
        )
    )
    word_count_df = pd.DataFrame.from_dict(dict( Counter(all_messages_combined_words_cleaned) ), orient = 'index', columns = ['count'])
    word_count_df = word_count_df.assign( 
        frequency = word_count_df['count'].apply(lambda x: x/len( all_messages_combined_words_cleaned )) 
    )
    word_count_df = word_count_df.sort_values('frequency', ascending=False).reset_index()
    word_count_df.rename(index = str, columns = {'index': 'word'}, inplace = True)

    return df, word_count_df

def clean_data(df):
    """Removes duplicates from the dataset
    Args:
    df: pandas.DataFrame: Input data containing messages and their classifications into multiple categories

    Returns:
    df: pandas.DataFrame: Deduplicated input data    
    """
    return df.drop_duplicates()

def save_data(df, table_name, database_filename):
    """Writes the dataframe into a sqlite database at the given location
    
    Args:
    df: pandas.DataFrame: Input data containing messages and their classifications into multiple categories    
    table_name: string. Table to write the input data frame to
    database_filename: File location to create and store SQLite database

    Returns:
    None
    
    """
    engine = create_engine('sqlite:///' + database_filename, echo  = True)
    df.to_sql(table_name, engine, index=False, if_exists='replace', chunksize=100 )

def main():
    """Main function for the file. This is entry point of execution
    
    Args:
    None

    Returns:
    None

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, word_count_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, 'messages', database_filepath)
        save_data(word_count_df, 'word_count', database_filepath)

        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()