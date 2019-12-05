from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split




moviedir = r'D:\Dokument\Skolan\TNM108-Machinelearning\TNM108-Machine-learning-for-social-media\Lab4\movie_reviews\movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)

len(movie.data)

# target names ("classes") are automatically generated from subfolder names
movie.target_names

# First file seems to be about a Schwarzenegger movie. 
movie.data[0][:500]

# first file is in "neg" folder
movie.filenames[0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
movie.target[0]

# Split data into training and test sets
''' Fråga om detta - hur ska vi dela upp datan, vill ju komma åt target_names'''
#movie_train, movie_test = train_test_split(movie, test_size = 0.20, random_state = 12)
moviedata_train, moviedata_test, movietarget_train, movietarget_test = train_test_split(movie.data, movie.target, 
                                                         test_size = 0.20, random_state = 12)



# training SVM classifier

text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])

text_clf.fit(moviedata_train, movietarget_train)


docs_test = moviedata_test
predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted == movietarget_test))


print(metrics.classification_report(movietarget_test, predicted, target_names=movie.target_names))

print(metrics.confusion_matrix(movietarget_test, predicted))


parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

gs_clf = gs_clf.fit(moviedata_train[:400], movietarget_train[:400])


# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']


print(movie.target_names[gs_clf.predict(reviews_new)[0]])

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


pred = gs_clf.predict(reviews_new)

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))
