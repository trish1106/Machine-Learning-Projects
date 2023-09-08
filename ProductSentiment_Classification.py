import turicreate as tc

# Reading the data
products = tc.Sframe('amazon_bavy.sframe')

# Building a word count vector for each review
products.groupby('name', operations= {'count': tc.aggregare.COUNT()}).sort('count', ascending = False)

# Building a word count vector
products['word_count'] = tc.text_analytics.count_words(products['review'])

# Defining positive and negative sentiment
# ignoring all 3* reviews
product = products[products['rating'] != 3]

# positive sentiment = 4* and 5*
products['sentiments'] = products['rating'] >= 4

# Building the sentiment classifier
def sentiment_classifier(x):
    train_data, test_data = products.random_split(0.8, seed = 0)

    model = tc.logist_classifier.create(train_data, target = 'sentiments', features = [x], validation_set = test_data)

    # evaluating the model using 'word_count' as feature
    model.evaluate()

    return model

sentiment_model = sentiment_classifier('word_count')


# Applying the learned model to understand the sentiment for giraffe
products['predicted_sentiment'] = sentiment_model.predict(products, output_type = 'probability')

# Function to examin the reviews for any product
def reviews(y):
    x = products[products['name'] == y]
    return x

#Evaluating the results of the model on the Giraffe products reviews

giraffe_reviews = reviews('Vulli Sophie the Giraffe Teether')

# On printing the head and tail of above review, we can compare the predicted sentiment for the actual sentiment
print(giraffe_reviews.head())
print(giraffe_reviews.tail())
