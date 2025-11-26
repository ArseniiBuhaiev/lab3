from nltk.corpus import movie_reviews, stopwords
from nltk import FreqDist

stop_words = stopwords.words(fileids="english")

# Створення списку з усіх слів, укладання частотного словника
all_words_uncleared = list(movie_reviews.words())
all_words = [w for w in all_words_uncleared if w.isalpha()
             and w not in stop_words]

words_freq = FreqDist(all_words)

# Створення списку зі слів у позитивних відгуках, 
# укладання частотного словника
pos_words = list(movie_reviews.words(categories="pos"))
pos_freq = FreqDist(pos_words)

# Створення списку зі слів у негативних відгуках,
# укладання частотного словника
neg_words = list(movie_reviews.words(categories="neg"))
neg_freq = FreqDist(neg_words)

# Створити список, у який додати всі слова з усіх відгуків,
# відсортувати його за частотою вживання і вивести в консоль
# 25 найбільш уживаних у цьому корпусі.
print("Найчастотніші слова у корпусі:")
for w in words_freq.most_common(25):
    print(f"\t{w[0]}\t{w[1]}")

# Знайти кількість вживань слова за індивідуальним варіантом:
# а) у корпусі;
# б) серед позитивних відгуків;
# в) серед негативних відгуків
print(f"\nЧастота слова wonderful:")
print(f"\tу корпусі\t{words_freq["wonderful"]}\
\n\tу позитивних відгуках\t{pos_freq["wonderful"]}\
\n\tу негативних відгуках\t{neg_freq["wonderful"]}")

def is_in_review(words: list, fileid: str, only_true=True) -> bool:
    result = {}
    review_words = set(movie_reviews.words(fileids=fileid))

    for word in words:
        result[word[0]] = word[0] in review_words

    if only_true:
        result = {word: present for word, present in result.items() if present}
        return result
    else:
        return result

print("\nФункція з перевірки наявности слова у відгуку:")
presence_result = is_in_review(words=words_freq.most_common(3200), fileid="pos/cv002_15918.txt")

for k, v in presence_result.items():
    print(f"\t{k}\t{v}")