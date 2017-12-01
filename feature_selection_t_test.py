import numpy as np
import sys, math
import scipy.stats

class NormalDistributionSampleGenerator:
    def __init__(self, mean, sd):
        self.MEAN = mean
        self.SD = sd
    def getSamples(self, n):
        return np.random.normal(self.MEAN, self.SD, n)
    def getProb(self, x):
        var = float(self.SD)**2
        pi = 3.1415926
        denom = (2*pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom
        

def help(code_file):
    print("Wrong Input! Sample Usage:\n python {0} {1}\n".format(code_file, 30))

if __name__=='__main__':
    if len(sys.argv) != 2:
        help(sys.argv[0])
    try:
        N = int(sys.argv[1])
    except:
        help(sys.argv[0])

    # Significance value
    significance_value = 0.05

    # Draw samples from two distributions
    D1 = NormalDistributionSampleGenerator(1,1).getSamples(N)
    D2 = NormalDistributionSampleGenerator(1.5,1).getSamples(N)

    '''
    H_0 = m_1 - m_2 = 0
    H_1 = m_1 - m_2 != 0
    '''

    # Define a test statictic q
    mean1 = sum(D1)/len(D1)
    mean2 = sum(D2)/len(D2)
    q = (mean1 - mean2) # divided by 1

    # Define D=N(0,1) so that p(q|H_0) is large
    # Since difference of means is 0 and classes have equal sd = 1
    D = scipy.stats.norm(0,1)
    
    # Find pr_of_error = P(q \in complement(D) | H_0)
    error_prob = 1-D.pdf(q)
    if error_prob < significance_value:
        print 'H_0 (i.e, m1 - m2 = 0) holds!'
    else:
        print 'H_1 (i.e, m1 - m2 != 0) holds!'
