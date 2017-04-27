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

        c_next = chars[i - 1]

        if c == 'I':
            total += 1
        elif c == 'V':
            total += 5
        elif c == 'X':
            total += 10
        elif c == 'L':
            total += 50
        elif c == 'C':
            total += 100
        elif c == 'D':
            total += 500
        elif c == 'L':
            total += 1000


def roman_reducer(prev, current_n):
    prev_n, total = prev

    value_of = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }

    if prev_n == '_':
        return current_n, value_of[current_n]

    prev_val = value_of[prev_n]
    current_val = value_of[current_n]

    if prev_val < current_val:
        current_val -= (prev_val * 2)

    return current_n, total + current_val


import functools


def roman2(numeral):
    return functools.reduce(roman_reducer, list(numeral), ('_', 0))[1]

# print (roman2('MCLXVII'))


def pop_count(num):
    total = 0
    for _ in range(32):
        total += (num & 1)
        num = num >> 1
    return total

print(pop_count(7))