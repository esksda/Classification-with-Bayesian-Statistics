# importing pachages
import numpy as np
import random
# reading train data
train_male_female_data = np.loadtxt("male_female_X_train.txt")
train_male_female_category = np.loadtxt("male_female_y_train.txt")
# seperating male and female and calculating probability
male = train_male_female_category[train_male_female_category == 0]
female = train_male_female_category[train_male_female_category == 1]
probability_male = len(male)/(len(male)+len(female))
probability_female = len(female)/(len(male)+len(female))
print(f"Probability of male is {round(probability_male,4)}")
print(f"Probability of female is {round(probability_female,4)}")
## male female height weight
male_height = train_male_female_data[train_male_female_category == 0][:,0]
female_height = train_male_female_data[train_male_female_category == 1][:,0]
male_weight = train_male_female_data[train_male_female_category == 0][:,1]
female_weight = train_male_female_data[train_male_female_category == 1][:,1]
# defined range given in instructions for histogram
height_range= [80,220]
weight_range = [30,180]
# hsitogram points of male height and female height
histogram_male_height, bin_male_height = np.histogram(male_height, bins = 10, range = height_range)
histogram_female_height, bin_female_height = np.histogram(female_height, bins = 10, range = height_range)
# histogram bin mid points
height_mid_point = []
for i in range(len(histogram_male_height)):
    points = (bin_male_height[i] + bin_male_height[i+1]) / 2
    height_mid_point.append(points)
height_mid_point = np.array(height_mid_point)
# reading test data
test_male_female_data = np.loadtxt("male_female_X_test.txt")
test_male_female_category = np.loadtxt("male_female_y_test.txt")

# values for bigger probability of male female given height
male_female_given_height = []
for i in range(len(test_male_female_category)):
    sub_bin_height = np.abs(height_mid_point - test_male_female_data[i][0])
    target_bin_index = np.argmin(sub_bin_height)
    # probability for height
    prob_of_height = (histogram_male_height[target_bin_index] + histogram_female_height[target_bin_index]) / len(train_male_female_category)
    # calculating probability of height given male
    prob_of_height_given_male = histogram_male_height[target_bin_index] / len(male)
    # probability of male given height
    probability_of_male_given_height = (prob_of_height_given_male * probability_male) / prob_of_height
    
    # calculating probability of height given female
    prob_of_height_given_female = histogram_female_height[target_bin_index] / len(female)
    # probability of male given height
    probability_of_female_given_height = (prob_of_height_given_female * probability_female) / prob_of_height

    if probability_of_female_given_height > probability_of_male_given_height:
        male_female_given_height.append(1)
    else:
        male_female_given_height.append(0)
male_female_given_height = np.array(male_female_given_height)
current_prediction_for_height = sum(male_female_given_height == test_male_female_category) / len(test_male_female_category)
accu_for_height = round((100 * current_prediction_for_height), 4)
print(f"Accuracy for height is {accu_for_height} percent")

## histogram points of weight and doing same for weight
histogram_male_weight, bin_male_weight = np.histogram(male_weight, bins = 10, range = weight_range)
histogram_female_weight, bin_female_weight = np.histogram(female_weight, bins = 10, range = weight_range)
weight_mid_point = []
for i in range(len(histogram_male_weight)):
    weight_points = (bin_male_weight[i] + bin_male_weight[i+1]) / 2
    weight_mid_point.append(weight_points)
weight_mid_point = np.array(weight_mid_point)
male_female_given_weight = []
for i in range(len(test_male_female_category)):
    sub_bin_weight = np.abs(weight_mid_point - test_male_female_data[i][1])
    target_weight_bin_index = np.argmin(sub_bin_weight)
    # probability for weight
    prob_of_weight = (histogram_male_weight[target_weight_bin_index] + histogram_female_weight[target_weight_bin_index]) / len(train_male_female_category)
    # calculating probability of weight given male
    prob_of_weight_given_male = histogram_male_weight[target_weight_bin_index] / len(male)
    # probability of male given height
    probability_of_male_given_weight = (prob_of_weight_given_male * probability_male) / prob_of_weight
    
    # calculating probability of height given female
    prob_of_weight_given_female = histogram_female_weight[target_weight_bin_index] / len(female)
    # probability of male given height
    probability_of_female_given_weight = (prob_of_weight_given_female * probability_female) / prob_of_weight

    if probability_of_female_given_weight > probability_of_male_given_weight:
        male_female_given_weight.append(1)
    else:
        male_female_given_weight.append(0)
male_female_given_weight = np.array(male_female_given_weight)
current_prediction_for_weight = sum(male_female_given_weight == test_male_female_category) / len(test_male_female_category)
accu_for_weight = round((100 * current_prediction_for_weight), 4)
print(f"Accuracu for weight is {accu_for_weight} percent")
# same for height and weight
male_female_given_height_weight = []
for i in range(len(test_male_female_category)):
    # male height and weight
    sub_bin_height = np.abs(height_mid_point - test_male_female_data[i][0])
    target_bin_index = np.argmin(sub_bin_height)
    # probability for height
    prob_of_height = (histogram_male_height[target_bin_index] + histogram_female_height[target_bin_index]) / len(train_male_female_category)
    # calculating probability of height given male
    prob_of_height_given_male = histogram_male_height[target_bin_index] / len(male)
    # probability of male given height
    probability_of_male_given_height = (prob_of_height_given_male * probability_male) / prob_of_height

    #---------------------

    sub_bin_weight = np.abs(weight_mid_point - test_male_female_data[i][1])
    target_weight_bin_index = np.argmin(sub_bin_weight)
    # probability for weight
    prob_of_weight = (histogram_male_weight[target_weight_bin_index] + histogram_female_weight[target_weight_bin_index]) / len(train_male_female_category)
    # calculating probability of weight given male
    prob_of_weight_given_male = histogram_male_weight[target_weight_bin_index] / len(male)
    # probability of male given weight
    probability_of_male_given_weight = (prob_of_weight_given_male * probability_male) / prob_of_weight

    #------------------
    probability_of_male_hw = probability_of_male_given_height * probability_of_male_given_weight

    # female height and weight
    sub_bin_height = np.abs(height_mid_point - test_male_female_data[i][0])
    target_bin_index = np.argmin(sub_bin_height)
    # probability for height
    prob_of_height = (histogram_male_height[target_bin_index] + histogram_female_height[target_bin_index]) / len(train_male_female_category)
    
    # calculating probability of height given female
    prob_of_height_given_female = histogram_female_height[target_bin_index] / len(female)
    # probability of female given height
    probability_of_female_given_height = (prob_of_height_given_female * probability_female) / prob_of_height

    #-------------
    sub_bin_weight = np.abs(weight_mid_point - test_male_female_data[i][1])
    target_weight_bin_index = np.argmin(sub_bin_weight)
    # probability for weight
    prob_of_weight = (histogram_male_weight[target_weight_bin_index] + histogram_female_weight[target_weight_bin_index]) / len(train_male_female_category)
    
    # calculating probability of height given female
    prob_of_weight_given_female = histogram_female_weight[target_weight_bin_index] / len(female)
    # probability of female given weight
    probability_of_female_given_weight = (prob_of_weight_given_female * probability_female) / prob_of_weight
    probability_of_female_hw = probability_of_female_given_height * probability_of_female_given_weight

    if probability_of_female_hw > probability_of_male_hw:
        male_female_given_height_weight.append(1)
    else:
        male_female_given_height_weight.append(0)
male_female_given_height_weight = np.array(male_female_given_height_weight)
classifier = (male_female_given_height_weight == test_male_female_category)
whole_accuracy = round(100 * (sum(classifier)/len(test_male_female_category)), 4)
print(f"Accuracy of height and weight is {whole_accuracy} percent")
## Commenting others
print("Other required probabilities used to calculate the required accuracied and others")
print(f"Probability of height given male is {prob_of_height_given_male}")
print(f"Probability of weight given male is {prob_of_weight_given_male}")
print(f"Probability of height given female is {prob_of_height_given_female}")
print(f"Probability of weight given female is {prob_of_weight_given_female}")