# print ('''hello world''')
# print("please enter a string : ")
# # name=input()
# # print(name)
# print(r'''hello,\n
# world''')
# print('%2d-%02d' % (3, 1))
# print('%.2f' % 3.1415926)
#
# print('Age: %s. Gender: %s' % (25, True))
# list
# array=[1,2,3,4,4]
# print(len(array))
# print(array[2])
# array.append(9)
# #print(array)
# array.insert(2,8)
# #print(array)
# array.pop(1)
# #print(array)
# array.pop(2)
# print(array)
# tuple
# leng= [
#     ['Apple', 'Google', 'Microsoft'],
#     ['Java', 'Python', 'Ruby', 'PHP'],
#     ['Adam', 'Bart', 'Lisa']
# ]
# print(leng[0][0])
# print(leng[1][1])
# print(leng[2][2]+'\n')
import math

age = 20
if age >= 6:
    print('teenager')
elif age >= 18:
    print('adult')
else:
    print('kid')

# birth = int(input('birth: '))
# if birth < 2000:
#     print('00前')
# else:
#     print('00后')

BMI = 80.5 * 80.5 / 175
print(BMI)
if BMI < 18.5:
    print("guoqing")
elif 18.5 <= BMI < 25:
    print("zhengchang")
elif 25 <= BMI < 28:
    print("guozhong")
elif 28 <= BMI < 32:
    print("feipang")
else:
    print("yanzhongfeipang")

sum1 = 0
for x in range(101):
    sum1 = sum1 + x
print(sum1)

array1 = ['bart', 'lisa', 'adam']
for name in array1:
    print(name)
# dict
dictionary = {'zjh': 99, 'hua': 98, 'asd': 97, 'qwe': 96}
print(dictionary['zjh'])

print('zjhhua' in dictionary)
print(dictionary.get('zjh'))
# print(dictionary.get('hua',-1))
print(dictionary.pop('qwe'))
print(dictionary)
dictionary['zjhhua'] = 100
print(dictionary.keys())
print(dictionary.values())
# set
SET = set([9, 8, 7])
SET.add(5)
SET.remove(7)
print(SET)

a = 'abc'
b = a.replace('a', 'A')
print(b)
print(a)

number = 100
number2 = 101
print(hex(number))


def my_abs(x):
    """

    :param x:
    :return:
    """
    if x >= 0:
        return x
    else:
        return -x


print(my_abs(-90))


def null():
    pass


def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny


print(move(30, 50, 10, math.pi / 7))


def power(x, n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s


print(power(2, 11))
print(power(2))


# 定义可变参数和定义一个list或tuple参数相比，仅仅在参数前面加了一个*号。
# 在函数内部，参数numbers接收到的是一个tuple，因此，函数代码完全不变。
# 但是，调用该函数时，可以传入任意个参数，包括0个参数
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum


print(calc(2, 3, 1, 4))


# print(calc(*SET))
# 关键字参数,关键字参数在函数内部自动组装为一个dict,**[param]
def person(name, age, **kw):
    """

    :param name:
    :param age:
    :param kw:
    """
    print('name:', name, 'age:', age, 'other:', kw)


print(person("zjh", 21, shixi="yonyou"))


# 命名关键字参数
# 和关键字参数**kw不同，命名关键字参数需要一个特殊分隔符*，*后面的参数被视为命名关键字参数。
def person1(name, age, *, city, job):
    print(name, age, city, job)


print(person1("zjh", 21, city="beijing", job="sofware"))


# 成可接收一个或多个数并计算乘积：
def product(*x):
    sum = 1
    for n in x:
        sum = sum * n
    return sum


print(product(5))
print(product(5, 6))
print(product(5, 6, 7))


# 递归函数
# def fact(n):
#     if n==1:
#         return 1
#     return n * fact(n - 1)

def fact(n):
    return fact_iter(n, 1)


def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)


print(fact(5))


# 移动汉诺塔
def move(n, a, b, c):
    if n == 1:
        print('move', a, '-->', c)
    else:
        move(n - 1, a, c, b)
        move(1, a, b, c)
        move(n - 1, b, a, c)


move(3, 'A', 'B', 'C')
# 切片,list[param:param]
print(array1[-2:-1])
print(array1[0:1])


def trim(s):
    if len(s) == 0:
        return s
    elif s[0] == ' ':
        return (trim(s[1:]))
    elif s[-1] == ' ':
        return (trim(s[:-1]))
    return s


# print(trim("hello "))
# print(trim("  hello"))
# print(trim("  hello "))
# print(trim("  hello  world "))
# print(trim(""))
# print(trim("  "))
if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')


def findMinAndMax(L):
    if L != []:
        min = L[0]
        max = L[0]
        for i in L:
            if (max < i):
                max = i
            if (min > i):
                min = i
        return (min, max)
    else:
        return (None, None)


if findMinAndMax([]) != (None, None):
    print('测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('测试失败!')
else:
    print('测试成功!')

s = [x * x for x in range(1, 11)]
print(s)

L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = []
for i in L1:
    if (isinstance(i, str)):
        L2.append(i)
print(L1)
L2 = [s.lower() for s in L2]
print(L2)
if L2 == ['hello', 'world', 'apple']:
    print('测试通过!')
else:
    print('测试失败!')


def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1
    return 'done'


print(fib(8))


# 杨辉三角
def triangles():
    L = [1]
    while True:
        yield L
        L = [1] + [L[i] + L[i + 1] for i in range(len(L) - 1)] + [1]


print(triangles())
n = 0
results = []
for t in triangles():
    print(t)
    results.append(t)
    n = n + 1
    if n == 10:
        break
if results == [
    [1],
    [1, 1],
    [1, 2, 1],
    [1, 3, 3, 1],
    [1, 4, 6, 4, 1],
    [1, 5, 10, 10, 5, 1],
    [1, 6, 15, 20, 15, 6, 1],
    [1, 7, 21, 35, 35, 21, 7, 1],
    [1, 8, 28, 56, 70, 56, 28, 8, 1],
    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
]:
    print('测试通过!')
else:
    print('测试失败!')

g = (x * x for x in range(10))
print(g)
print(next(g))

fun = abs
print(fun(-10))


def add(x, y, fun):
    return fun(x) + fun(y)


print(add(-1, -9, abs))


##  高阶函数
def f(x):
    return x * x


r1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
r = map(f, r1)
print(list(r))
print(list(map(str, r1)))

L = []
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    L.append(f(n))
print(L)
print(sum(r1))
#  把str转换为int的函数：
from functools import reduce


def fn(x, y):
    return x * 10 + y


def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]


print(reduce(fn, map(char2num, '0123456789')))

DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


def str2int(s):
    def fn(x, y):
        return x * 10 + y

    def char2num(s):
        return DIGITS[s]

    return reduce(fn, map(char2num, s))


print(str2int('111223333'))


# 利用map()函数，把用户输入的不规范的英文名字，变为首字母大写，其他小写的规范名字
def normalize(name):
    name = name[0].upper() + name[1:].lower()
    return name


L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)


# 编写一个prod()函数，可以接受一个list并利用reduce()求积
def prod(L):
    def func(y, x):
        return y * x

    return reduce(func, L)


print(prod([3, 5, 7, 9]))
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
if prod([3, 5, 7, 9]) == 945:
    print('测试成功!')
else:
    print('测试失败!')


# 利用map和reduce编写一个str2float函数，把字符串转换成浮点数
# def str2float(s):
#
#     return s

# 用filter求素数
def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n


def _not_divisible(n):
    return lambda x: x % n > 0


def primes():
    yield 2
    it = _odd_iter()  # 初始序列
    while True:
        n = next(it)  # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it)  # 构造新序列


# 打印1000以内的素数:
for n in primes():
    if n < 100:
        print(n)
    else:
        break


# 利用filter()筛选出回数
def is_palindrome(n):
    # return n == int(str(n)[::-1])
    s = str(n)

    for i in list(range((len(s)) // 2)):
        if s[i] != s[-(i + 1)]:
            return False
    return True


output = filter(is_palindrome, range(1, 1000))
print('1~1000:', list(output))
if list(filter(is_palindrome, range(1, 200))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101,
                                                  111, 121, 131, 141, 151, 161, 171, 181, 191]:
    print('测试成功!')
else:
    print('测试失败!')

# 假设我们用一组tuple表示学生名字和成绩,分别按名字,成绩从高到低排序
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]


def by_name(t):
    return t[0].upper()


def by_score(s):
    return s[1]


L2 = sorted(L, key=by_name)
print(L2)
L3 = sorted(L, key=by_score, reverse=True)
print(L3)


# 闭包
def count():
    fs = []

    def f(n):
        def j():
            return n * n

        return j

    for i in range(1, 4):
        fs.append(f(i))
    return fs


f1, f2, f3 = count()
print(f1())
print(f2())
print(f3())


# 利用闭包返回一个计数器函数，每次调用它返回递增整数
def createCounter():
    F_ADD = []

    def counter():
        x = 0
        while True:
            x += 1
            yield x

    it = counter()

    def number():
        return next(it)

    return number


counterA = createCounter()
print(counterA(), counterA(), counterA(), counterA(), counterA())  # 1 2 3 4 5
counterB = createCounter()
if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:
    print('测试通过!')
else:
    print('测试失败!')


def is_odd(n):
    return n % 2 == 1


print(list(filter(is_odd, range(1, 20))))
# ||
# ||
# \/
print(list(filter(lambda n: n % 2 == 1, range(1, 20))))

# decorator 装饰器
import functools, time, datetime


def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)

        return wrapper

    return decorator


@log('excute')
def now():
    print(datetime.datetime.now())


print(now())


# 设计一个decorator，它可作用于任何函数上，并打印该函数的执行时间
def metric(fn):
    print('%s executed in %s ms' % (fn.__name__, 10.24))
    return fn


@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;


@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;


f = fast(11, 22)
s = slow(11, 22, 33)
if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')
else:
    print('success!!!')

######面向对象编程
std1 = {'name': 'Michael', 'score': 98}
std2 = {'name': 'Bob', 'score': 81}


def print_score(std):
    print('%s: %s' % (std['name'], std['score']))


print(print_score(std2))


## student class
class Student(object):

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print('%s: %s' % (self.name, self.score))


bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
bart.print_score()
lisa.print_score()


class Student1(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

    def get_gender(self):
        return self.gender

    def set_gender(self, gender):
        self.gender = gender


# 测试:
bart = Student1('Bart', 'male')
if bart.get_gender() != 'male':
    print('测试失败!')
else:
    bart.set_gender('female')
    if bart.get_gender() != 'female':
        print('测试失败!')
    else:
        print('测试成功!')


##继承和多态
class Animal(object):
    def run(self):
        print('Animal is running...')


class Dog(Animal):
    def run(self):
        print('Dog is running...')

    def eat(self):
        print('Eating meat...')


class Cat(Animal):
    def run(self):
        print('Cat is running...')


def run_twice(animal):
    animal.run()
    animal.run()


print(run_twice(Animal()))
print(run_twice(Dog()))


class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly...')


print(run_twice(Tortoise()))

print(type(run_twice))
print(dir(Student))
print(dir(Animal))


class Student(object):
    count = 0

    def __init__(self, name):
        Student.count = Student.count + 1
        self.name = name


# 测试:
if Student.count != 0:
    print('测试失败!')
else:
    bart = Student('Bart')
    if Student.count != 1:
        print('测试失败!')
    else:
        lisa = Student('Bart')
        if Student.count != 2:
            print('测试失败!')
        else:
            print('Students:', Student.count)
            print('测试通过!')


# __slots__
class Student(object):
    __slots__ = ('name', 'age')  # 用tuple定义允许绑定的属性名称


class GraduateStudent(Student):
    pass


s = Student()  # 创建新的实例
s.name = 'Michael'  # 绑定属性'name'
s.age = 25  # 绑定属性'age'
# ERROR: AttributeError: 'Student' object has no attribute 'score'
try:
    s.score = 99
except AttributeError as e:
    print('AttributeError:', e)

g = GraduateStudent()
g.score = 99
print('g.score =', g.score)


class Student(object):

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value


S = Student()
S.score = 67


# Screen
class Screen(object):
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @width.setter
    def height(self, value):
        self._height = value

    @property
    def resolution(self):
        return self._height * self._width


# 测试:
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')


# 定制类
class Student(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Student object (name=%s)' % self.name

    __repr__ = __str__


print(Student('Michael'))


# 无论API怎么变，SDK都可以根据URL实现完全动态的调用
class Chain(object):

    def __init__(self, path=''):
        self._path = path

    def __getattr__(self, path):
        return Chain('%s/%s' % (self._path, path))

    def __str__(self):
        return self._path

    __repr__ = __str__


print(Chain().status.user.timeline.list)

# 使用枚举类
from enum import Enum, unique

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr',
                       'May', 'Jun', 'Jul', 'Aug',
                       'Sep', 'Oct', 'Nov', 'Dec'))
for name, member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)


@unique
class Weekday(Enum):
    Sun = 0  # Sun的value被设定为0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6


print(Weekday.Thu.value)


##把Student的gender属性改造为枚举类型，可以避免使用字符串
class Gender(Enum):
    Male = 0
    Female = 1


class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender


# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')


###使用元类
# 要创建一个class对象，type()函数依次传入3个参数：
# class的名称；
# 继承的父类集合，注意Python支持多重继承，如果只有一个父类，别忘了tuple的单元素写法；
# class的方法名称与函数绑定，这里我们把函数fn绑定到方法名hello上。


######   metaclass
# metaclass是类的模板，所以必须从`type`类型派生：
class ListMetaclass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)


class MyList(list, metaclass=ListMetaclass):
    pass


mylist = MyList()
mylist.add(1)
print(mylist)

##  错误处理
#####try....except....finally...

from functools import reduce


def str2num(s):
    # return int(s)
    try:
        number = int(s)
        # 注意这个地方的写法 是在except里面再写一个try except 而不是并且的写except！
        # 有点类似于if else 里面的嵌套if else
    except:
        try:
            number = float(s)
        except:
            print("what you input is not a number!!!")

    finally:

        return number


def calc(exp):
    ss = exp.split('+')
    ns = map(str2num, ss)
    return reduce(lambda acc, x: acc + x, ns)


def main():
    r = calc('100 + 200 + 345')
    print('100 + 200 + 345 =', r)
    r = calc('99 + 88 + 7.6')
    print('99 + 88 + 7.6 =', r)


main()


#####对函数fact(n)编写doctest并执行
def fact(n):
    '''
       Function to get n!
       Example:
       >>> fact(1)
       1
       >>> fact(2)
       2
       >>> fact(3)
       6
       >>> fact('a')
       Traceback(most recent call last)
           ...
       KeyError: 'a'
       '''
    if n < 1:
        raise ValueError()
    if n == 1:
        return 1
    return n * fact(n - 1)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
print(fact(5))
##I/O编程
with open('/home/zjh/nohup.out', 'rb') as f:
    print(f.read())
import os

print(os.uname())
print(os.environ)
##序列化
import json

d = dict(name='Bob', age=20, score=88)
data = json.dumps(d)
print('JSON Data is a str:', data)
reborn = json.loads(data)
print(reborn)

class Student(object):

    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

    def __str__(self):
        return 'Student object (%s, %s, %s)' % (self.name, self.age, self.score)

s = Student('Bob', 20, 88)
std_data = json.dumps(s, default=lambda obj: obj.__dict__)
print('Dump Student:', std_data)
rebuild = json.loads(std_data, object_hook=lambda d: Student(d['name'], d['age'], d['score']))
print(rebuild)
##对中文进行JSON序列化
obj = dict(name='小明', age=20)
s = json.dumps(obj, ensure_ascii=True)
print(s)
##多进程
print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
from multiprocessing import Process

# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
from multiprocessing import Pool,Queue
import os, time, random,threading

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

import subprocess

print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)

print('$ nslookup')
p = subprocess.Popen(['nslookup'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, err = p.communicate(b'set q=mx\npython.org\nexit\n')
print(output.decode('utf-8'))
print('Exit code:', p.returncode)


# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()

####多线程
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n=0
    while n<5:
        n=n+1
        print('thread %s >>> %s' % (threading.current_thread().name,n))
        time.sleep(1)
    print('thread %s is ended...' % threading.current_thread().name)

print('thread %s is running...' % threading.current_thread().name)
t=threading.Thread(target=loop,name='LOopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)

####Lock
balance = 0

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

lock = threading.Lock()

def run_thread(n):
    for i in range(100000):
        # 先要获取锁:
        lock.acquire()
        try:
            # 放心地改吧:
            change_it(n)
        finally:
            # 改完了一定要释放锁:
            lock.release()
t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
##############ThreadLocal
local_school = threading.local()

def process_student():
    # 获取当前线程关联的student:
    std = local_school.student
    print('Hello, %s (in student %s)' % (std, threading.current_thread().name))
def process_teacher():
    STD=local_school.teacher
    print('Hello, %s (in teacher %s)' % (STD, threading.current_thread().name))
def process_thread(name):
    # 绑定ThreadLocal的student:
    local_school.student = name
    process_student()
    local_school.teacher = name
    process_teacher()

t1 = threading.Thread(target= process_thread, args=('Alice',), name='Thread-A')
t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
t3 = threading.Thread(target= process_thread, args=('zjh',), name='Thread-C')
t4 = threading.Thread(target= process_thread, args=('hua',), name='Thread-D')
t1.start()
t2.start()
t3.start()
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()

#####常用内建模块
from  datetime import datetime
##datetime==>timestamp
dt = datetime(2015, 4, 19, 12, 20)
print(dt.timestamp())
##timestamp转换为datetime
t = 1429417200.0
print(datetime.fromtimestamp(t))
print('\n')
##str转换为datetime
cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday)
##datetime==>str
now=datetime.now()
print(now.strftime('%Y-%m-%d %H:%M:%S'))

#######hashlib
import hashlib
db = {
    'michael': 'e10adc3949ba59abbe56e057f20f883e',
    'bob': '878ef96e86145580c38c87f0410ad153',
    'alice': '99b1c2188db85afee403b1536010c2c9'
}
def login(user, password):
    # MD5=hashlib.md5()
    # if MD5.update(user.encode('utf-8'))==db.keys() \
    #         and MD5.update(password.encode('utf-8'))==db.values():
    #     return True
    # else:
    #     return False
    if user not in db:
        print
        'You have not Signed Up.'
        return False
    else:
        md5 = hashlib.md5()
        md5.update(password.encode('utf-8'))
        if md5.hexdigest() == db[user]:
            print
            'Old user'
            return True
        else:
            print
            'Wrong password'
            return False

# 测试:
assert login('michael', '123456')
assert login('bob', 'abc999')
assert login('alice', 'alice2008')
assert not login('michael', '1234567')
assert not login('bob', '123456')
assert not login('alice', 'Alice2008')
print('ok')
hello=('hello')

