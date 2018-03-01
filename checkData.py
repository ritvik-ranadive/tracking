import os
import numpy as np
from random import randint

script_dir = os.path.dirname(__file__)
test = open(os.path.join(script_dir, "test.txt"), 'r').read().split("\n")
train = open(os.path.join(script_dir, "train.txt"), 'r').read().split("\n")
db = open(os.path.join(script_dir, "db.txt"), 'r').read().split("\n")

random1 = [randint(0,3535) for p in range(0,10)]
random2 = [randint(0,7410) for p in range(0,10)]
random3 = [randint(0,1335) for p in range(0,10)]
print(random1)
print(random2)
print(random3)

test_random = [test[i] for i in random1]
train_random = [train[i] for i in random2]
db_random = [db[i] for i in random3]

print(test_random)
print(train_random)
print(db_random)
