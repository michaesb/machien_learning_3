import numpy as np
import matplotlib.pyplot as plt

"""
As we progressed we saw that we wanted to plot in the same way for a lot of the
methods, so we decided to create a generalized script.
"""

def plotting_ratio(ratio_, acc_score, prec_score, rec_score, title_method):
    # plt.plot(ratio_,acc_score, "-o")
    # plt.plot(ratio_,prec_score, "-o")
    # plt.plot(ratio_,rec_score, "-o")
    plt.semilogx(ratio_,acc_score, "-o")
    plt.semilogx(ratio_,prec_score, "-o")
    plt.semilogx(ratio_,rec_score, "-o")
    plt.xlabel("ratio")
    plt.ylabel("score")
    plt.title("Scikit-Learn "+ title_method +" score for different ratios")
    plt.legend(['accuracy', "precision", "recall"])
    plt.show()
