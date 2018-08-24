# -*- coding: utf-8 -*-
__author__="zjh"
#将一个字符串中的空格替换为 "ab"
def trim(s):
    string=""
    for i in range(len(s)):
        if s[i] == " ":
            string += "ab"
        else:
            string += s[i]
    return string
def trim1(s):
    return s.replace(" ","ab")
print(trim(" hello world "))
print(trim1("hello world"))



