
continue1 = True
x = 5
while continue1:
    print(f"current x is {x}")
    if x > 10:
        continue1 = False
    x += 1
    print(f"new x is {x}")