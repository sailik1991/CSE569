import numpy as np
import scipy.stats

if __name__ == '__main__':

    significance_value = 0.05

    for n in [30, 150, 600]:
        # Draw samples from two distributions
        D1 = np.random.normal(1,1,n)
        D2 = np.random.normal(1.5,1,n)

        t_score = scipy.stats.ttest_ind(D1, D2)

        if t_score > significance_value:
            print("Cannot Reject Null Hypothesis of identical avg. score.")
        else:
            print("Reject Null Hypothesis of identical avg. score")
