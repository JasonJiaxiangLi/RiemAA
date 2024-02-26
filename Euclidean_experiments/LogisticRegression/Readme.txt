This is the code for the logistic regression. It is based on the RNA code released on https://github.com/windows7lover/RegularizedNonlinearAcceleration. 

We use the covertype and sido0 datasets, which can be downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html and http://www.causality.inf.ethz.ch/data/SIDO.html, respectively.

We further scaled the matrix $A$ used in the logistic regression function, by using A=A/max(abs(A)). Our implementation of LM-AA can be found in LM_AA.m.

For the RNA source code, we have modified it in order to avoid numerical errors. Our modification can be found in adaptive_rna.m line 144 to line 147. Without these modifications, the RNA program would stop before it fully converges. We have not modified other parts of the RNA source code.

The two datasets have been included in the "datasets" folder. You only need to run comparison.m then you to generate figures and data in the current folder. The figures shown in the paper have a shorter range of computational time in order to improve readability. You can use processing_figures.m to generate the figures used in the paper.