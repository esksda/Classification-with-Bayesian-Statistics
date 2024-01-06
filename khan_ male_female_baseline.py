import numpy as np
import random
test_data = np.loadtxt("male_female_y_test.txt")
empty_list = []
for i in range(len(test_data)):
    value = random.randint(0,1)
    empty_list.append(value)
random_number = np.array(empty_list)
random_accuracy_checker = (test_data == random_number)
accuracy = round(100 * (sum(random_accuracy_checker)/len(random_accuracy_checker)), 4)
print(f"Random accuracy is {accuracy} percent")

train_data = np.loadtxt("male_female_y_train.txt")
number_of_male = train_data[train_data == 0]
number_of_female = train_data[train_data == 1]
if len(number_of_female) > len(number_of_male):
    print("Take female as likelihood")
else:
    print("Take male as likelihood")
list_for_test_data = []
for i in range(len(test_data)):
    list_for_test_data.append(1)
np_list_for_test_data = np.array(list_for_test_data)
likelihood_accuracy_checker = (test_data == np_list_for_test_data)
likelihood_accuracy = round(100 * (sum(likelihood_accuracy_checker)/len(likelihood_accuracy_checker)),4)
print(f"Accuracy is {likelihood_accuracy} percent")

