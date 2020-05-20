import math
def make_in_bounds(x):
    ok = False
    while not ok:
        if x<=math.pi and x>-math.pi:
            ok = True
        else:
            if x>math.pi:
                x = x - (2 * math.pi)
            elif x<=-math.pi:
                x = x + (2 * math.pi)
    return x

print(make_in_bounds(-7*math.pi))