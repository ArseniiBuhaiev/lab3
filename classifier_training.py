import random
import pickle
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import accuracy

from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB

stop_words = stopwords.words(fileids="english")

# Формування датасету для навчання та тестування
def get_features(words_list: list) -> bool:
    words_list = [w for w in words_list if w not in stop_words and w.isalpha()]
    return dict([(word, True) for word in words_list])

reviews = [(get_features(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(reviews)
training_set = reviews[:1700]
testing_set = reviews[1700:]

# Тренування моделі через NLTK
nltk_classifier = NaiveBayesClassifier.train(training_set)
nltk_accuracy = accuracy(nltk_classifier, testing_set) * 100

nltk_classifier.show_most_informative_features(25)

with open("models/NB_classifier.pickle", "wb") as model:
    pickle.dump(nltk_classifier, model)

# Тренування моделі через scikit-learn
classifier_skl = SklearnClassifier(BernoulliNB()).train(training_set)
skl_accuracy = accuracy(classifier_skl, testing_set) * 100

with open("models/BernoulliNB_classifier.pickle", "wb") as model:
    pickle.dump(classifier_skl, model)

# Оцінки точности
print(f"\n\tNaive Bayes Algorithm accuracy percent: {nltk_accuracy:.2f}%")
print(f"\tBernoulli Naive Bayes Algorithm accuracy percent: {skl_accuracy:.2f}%\n")

best_accuracy = max(nltk_accuracy, skl_accuracy)

if best_accuracy == nltk_accuracy:
    print(f"Найкращий алгоритм: Naive Bayes (nltk) - {best_accuracy:.2f}% accuracy\n")
else:
    print(f"Найкращий алгоритм: BernoulliNB (scikit-learn) - {best_accuracy:.2f}% accuracy\n")

with open("best_NB.pickle", "rb") as model:
    nb = pickle.load(model)

with open("best_BernoulliNB.pickle", "rb") as model:
    bnb = pickle.load(model)

print(f"Найкращий результат Naive Bayes: {accuracy(nb, training_set) * 100:.2f}%")
print(f"Найкращий результат BernoulliNB: {accuracy(bnb, training_set) * 100:.2f}%")