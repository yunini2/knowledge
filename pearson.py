import math
def pearson(vector1, vector2):
    n = len(vector1)
    # simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    # sum up the square
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    # sum up the products
    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
    rho = (p_sum - (sum1 * sum2/n)) / math.sqrt((sum1_pow - pow(sum1, 2)/n) * (sum2_pow - pow(sum2, 2)/n))
    if rho == 0.0:
        return 0.0
    return rho
pearson(list(data['xx']), listdata['xx'])) # 0.9981191651097102
pearson(list(data['xx']), list(data['xxx'])) # -0.07462375694166053
