import turicreate as tc
import matplotlib as plt
%matplotlib inline

# Loading the data frame
sales = tc.SFrame('home_data.sframe')

# Exploring the data
# Comparing the corelation between size of the house and price
tc.show(sales[1:5000]['sqft_living'], sales[1:5000]['price'])

# splitting the data
training_set, test_set = sales.random_split(.8, seed = 0)

# Building the regression model
def reg_model():

    model = tc.linear_regression.create(training_set, target = 'price', feature = ['sqft_living'])

    return model

sqft_model = reg_model()

# Evaluating the model
print(sqft_model.evaluate(test_set))

# Visualize the predictions
plt.plot(test_set['sqfy_living'], test_set['price'], '.', test_set['sqft_living'], sqft_model.predict(test_set), '-')

# Exploring other features

my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']

# Building a regression model only based on my_features

my_features_model = tc.linear_regression.create(training_set, target = 'price', features = my_features)

# Evaluating my_features_model and comparing to the previous model which uses all the features

print(sqft_model.evaluate(test_set))
print(my_features_model.evaluate(test_set))


