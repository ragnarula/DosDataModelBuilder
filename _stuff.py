from itertools import chain


def generator1():
    for i in range(10):
        yield i


def generator2(i):
    for j in range(10):
        yield i + j


def flatmap(f, items):
    return chain.from_iterable(map(f, items))

# print(list(flatmap(generator2, generator1())))


def roman(string):
    chars = list(string.upper())
    total = 0
    for i in range(len(chars)):

        c = chars[i]

        if c == 'I':
            total += 1
            if


roman('one')