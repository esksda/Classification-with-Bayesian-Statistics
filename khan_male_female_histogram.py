import numpy as np
import matplotlib.pyplot as plt
import random
mf_hw = np.loadtxt("male_female_X_train.txt")
mfheight = mf_hw[:,0]
mfweight = mf_hw[:,1]
mf_class = np.loadtxt("male_female_y_train.txt")
male_height = mfheight[mf_class == 0]
female_height = mfheight[mf_class == 1]
male_weight = mfweight[mf_class == 0]
female_weight = mfweight[mf_class == 1]
hb_range = [80,220]
wb_range = [30,180]
## histogram using numpy
histogram_male_height, bin_male_height = np.histogram(male_height, bins = 10, range = hb_range)
histogram_female_height, bin_female_height = np.histogram(female_height, bins = 10, range = hb_range)
histogram_male_weight, bin_male_weight = np.histogram(male_weight, bins = 10, range = wb_range)
histogram_female_weight, bin_female_weight = np.histogram(female_weight, bins = 10, range = wb_range)

## plotting 
plt.figure()
plt.hist(male_height, bins = 10, range = hb_range, alpha = 0.5, label = 'Male', color = 'red', edgecolor = 'black')
plt.hist(female_height, bins = 10, range = hb_range, alpha = 0.5, label = 'Female', color = 'green', edgecolor = 'black')
plt.legend()
plt.figure()
plt.hist(male_weight, bins = 10, range = wb_range, alpha = 0.5, label = 'Male', color = 'red', edgecolor = 'black')
plt.hist(female_weight, bins = 10, range = wb_range, alpha = 0.5, label = 'Female', color = 'green', edgecolor = 'black')
plt.legend()
plt.show()
