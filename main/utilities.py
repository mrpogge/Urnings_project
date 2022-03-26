import numpy as np
import scipy.stats as sp
import scipy.special as sps

def binomial_gof(array, urn_size, true_p):
    a_len = len(array)
    elements_count = {}
    # iterating over the elements for frequency
    for element in array:
        # checking whether it is in the dict or not
        if element in elements_count:
            # incerementing the count by 1
            elements_count[element] += 1
        else:
            # setting the count to 1
            elements_count[element] = 1
    
    for i in range(urn_size):
        if i not in elements_count.keys():
            elements_count[i] = 0
    
    #the number of unique elements
    n = len(elements_count)
    khi_sq = 0

    for ky in range(len(elements_count)):
        x = list(elements_count.keys())[ky]
        print(x)
        expected_count = (sps.binom(urn_size + 1,x) * (true_p ** x) * ((1-true_p) ** (urn_size-x))) * a_len
        print(expected_count, elements_count[ky])

        #calculating test statistics
        khi_sq = khi_sq + ((elements_count[ky] - expected_count) ** 2 / expected_count)

    p = 1 - sp.chi2.cdf(khi_sq, n-1)
    return khi_sq, p

def frequency_check(array, set):
    elements_count = {}
    # iterating over the elements for frequency
    for element in array:
        # checking whether it is in the dict or not
        if element in elements_count:
            # incerementing the count by 1
            elements_count[element] += 1
        else:
            # setting the count to 1
            elements_count[element] = 1
    
    return elements_count
