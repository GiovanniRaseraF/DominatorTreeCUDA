
import math
maxScale = 100000
val = 2
power = 1.2

for i in range(1, 30):
    val = (math.ceil(val*power))
    print(f"make threads=1024 blocks=1024 scalar={val} OPTIM=3")
