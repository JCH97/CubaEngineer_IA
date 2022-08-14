import random
from typing import List, Tuple

from numpy import array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

LEN_IN_BITS = 10


def get_features(numb: int) -> List:
    return [numb >> i & 1 for i in range(LEN_IN_BITS)]


def build(n: int = 1 << 5) -> Tuple[List[int], List[List[int]]]:
    x = []
    y = []

    s = set()
    for _ in range(n):
        numb = random.randint(100, 1 << LEN_IN_BITS)
        s.add(numb)

    for numb in s:
        x.append(get_features(numb))
        if numb % 15 == 0:
            y.append(3)
        elif numb % 5 == 0:
            y.append(2)
        elif numb % 3 == 0:
            y.append(1)
        else:
            y.append(4)

    return array(x), array(y)


def map(ans: int) -> str:
    if ans == 1:
        return "Fizz"
    elif ans == 2:
        return "Buzz"
    elif ans == 3:
        return "FizzBuzz"
    else:
        return "None"


if __name__ == "__main__":
    trX, trY = build()

    model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    x_train, x_test, y_train, y_test = train_test_split(trX, trY, test_size=0.33, random_state=42)

    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_predictions)}')

    # for i in range(100):
    #     print(f'{i}: {map(model.predict([get_features(i)])[0])}')

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(trX):
        x_train, x_test = trX[train_index], trX[test_index]
        y_train, y_test = trY[train_index], trY[test_index]

        m = LogisticRegression()
        m.fit(x_train, y_train)
        y_predictions = model.predict(x_test)
        scores.append(accuracy_score(y_test, y_predictions))

    print(f'Accuracy: {sum(scores) / len(scores)}')
    print(scores)
