import random

NUM_OF_HATS = 3
TRIES = 10000

hats = [x for x in range(NUM_OF_HATS)]
switch = 0.0
no_switch = 0.0

for i in range(TRIES):
    prize = random.randint(0,NUM_OF_HATS-1)
    choose = random.randint(0,NUM_OF_HATS-1)
    if prize == choose:
        no_switch += 1.0
    if prize != choose:
        switch += 1.0

total = switch+no_switch
switch /= total
no_switch /= total
print("Switch: {}\nNot Switch: {}".format(switch, no_switch))
