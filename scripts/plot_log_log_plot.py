import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

file1 = np.load("penn_rnn2_corr0.5_v3timing.npz")
file2 = np.load("penn_rnn2_adadeltatiming.npz")
file3 = np.load("penn_rnn2_sgd_momtiming.npz")

#import ipdb; ipdb.set_trace()
#vals1 = [49.0] + file1['validppl'][np.where(file1['validppl']!=0)[0]].tolist()
#vals2 = [50.1] + file2['validppl'][np.where(file2['validppl']!=0)[0]].tolist()
vals1 = file1['validppl'][np.where(file1['validppl']!=0)[0]].tolist()
vals2 = file2['validppl'][np.where(file2['validppl']!=0)[0]].tolist()
vals3 = file3['validppl'][np.where(file2['validppl']!=0)[0]].tolist()

#plt.gca().set_yscale("log")

plt.plot(vals1)
plt.plot(vals2)
plt.plot(vals3)

plt.xlabel('Each Number of 10 updates')
plt.ylabel('validation perplexity')
plt.legend(["curveprop", "adadelta", "sgd+momentum"])
plt.savefig("adaptive_curvature_prop.png")
#show()
