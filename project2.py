import csv
csv.register_dialect('myDialect', delimiter=' ')
import copy
import math
from sklearn.svm import SVC

def loadDataset(filename):
    with open(filename, "rt") as csvfile:
        lines = csv.reader(csvfile, dialect='myDialect')
        data = []
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(len(dataset[x])):
                try:
                    dataset[x][y] = int(dataset[x][y])
                except:
                    try:
                        dataset[x][y] = float(dataset[x][y])
                    except:
                        dataset[x][y] = dataset[x][y]
    return dataset


# In the subpartone i.e, project 2 task a
# we have to write a sub routine which computes the distance between two data instances based on their category (numerical or categorical so called string type data.)
def distance(training_instance, test_instance):
    # findDistanceBetweenTwoInstances
    dist = 0  # Initial distance
    for ind in range(len(test_instance)):
        if (type(training_instance[ind]) == str):
            # Below distance is calculated for categorical values, so called Hamming distance
            dist = dist + computeHammingDistance(training_instance[ind], test_instance[ind])
        else:
            # Below distance is calculated for numerical values, so called Euclidean distance
            dist = dist + computeEuclideanDistance(training_instance[ind], test_instance[ind])
    return dist  # final distance between two instances


def computeHammingDistance(str1, str2):
    # Assuming that the both strings were different before starting comparisions so the distance should be maximum length string
    maxPossibleDist = len(str1) if len(str1) > len(str2) else len(str2)
    minLengthStr = len(str1) if len(str1) < len(str2) else len(str2)
    for ind in range(minLengthStr):
        # If any of character matches then we will decreament our count of overall distance (here, overall distance in the beginning is: assumed max length integer from given two strings)
        if (str1[ind] == str2[ind]):
            maxPossibleDist = maxPossibleDist - 1
    return maxPossibleDist


def computeEuclideanDistance(point1, point2):
    return ((point1 - point2) ** 2)


def KNN(training_set, test_set, k):
    for train_inst in training_set:
        train_inst.append(distance(train_inst, test_set))
    sorted_training_inst = sortTrainingSetToCheckDistance(training_set)
    #DONE till now
    flag = 0
    if (k < len(sorted_training_inst) and sorted_training_inst[k - 1][len(sorted_training_inst[0]) - 1] ==
            sorted_training_inst[k][len(sorted_training_inst[0]) - 1]):
        dist_less_then_last_conflicting_dists = [ind for ind in range(k) if
                                                 sorted_training_inst[ind][len(sorted_training_inst[0]) - 1] <
                                                 sorted_training_inst[k - 1][len(sorted_training_inst[0]) - 1]]
        wightedValueK = k - len(dist_less_then_last_conflicting_dists) # [0,1,2,3,4]
        conflicting_dist = [ind for ind in range(len(sorted_training_inst)) if
                            sorted_training_inst[ind][len(sorted_training_inst[0]) - 1] == sorted_training_inst[k - 1][
                                len(sorted_training_inst[0]) - 1]]
        repeating_dist_at_k = len(conflicting_dist) # [5,6]
        dictionary_for_identifying_feature = {}
        for index in range(len(dist_less_then_last_conflicting_dists)):
            if (sorted_training_inst[index][len(sorted_training_inst[0]) - 2] in dictionary_for_identifying_feature):
                dictionary_for_identifying_feature[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] = \
                dictionary_for_identifying_feature[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] + 1
            else:
                dictionary_for_identifying_feature[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] = 1
        for index in range(len(dist_less_then_last_conflicting_dists), len(dist_less_then_last_conflicting_dists) + len(conflicting_dist)):
            if (sorted_training_inst[index][len(sorted_training_inst[0]) - 2] in dictionary_for_identifying_feature):
                dictionary_for_identifying_feature[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] = \
                dictionary_for_identifying_feature[
                    sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] + float(
                    (wightedValueK) / repeating_dist_at_k)
            else:
                dictionary_for_identifying_feature[
                    sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] = float(
                    (wightedValueK) / repeating_dist_at_k)
        flag = 1
    if (flag == 1):
        predicted_class = max(dictionary_for_identifying_feature, key=dictionary_for_identifying_feature.get)
    else:
        feature_count_dictionary = {}
        for index in range(k):
            if (sorted_training_inst[index][len(sorted_training_inst[0]) - 2] in feature_count_dictionary):
                feature_count_dictionary[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] = \
                feature_count_dictionary[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] + 1
            else:
                feature_count_dictionary[sorted_training_inst[index][len(sorted_training_inst[0]) - 2]] = 1
        predicted_class = max(feature_count_dictionary, key=feature_count_dictionary.get)
    return predicted_class


def sortTrainingSetToCheckDistance(training_set_to_sort):
    indexOfDistance = len(training_set_to_sort[0]) - 1
    training_set_to_sort.sort(key=lambda x: x[indexOfDistance])
    return training_set_to_sort


def dropUnnecessaryFeatures(train_data, test_data):
    removableFeatureIndex = []
    for ind in range(len(train_data[0])):
        if (train_data[0][ind] == "fnlwgt"):
            removableFeatureIndex.append(ind)
        if (train_data[0][ind] == "EducationNum"):
            removableFeatureIndex.append(ind)
        if (train_data[0][ind] == "CapitalGain"):
            removableFeatureIndex.append(ind)
        if (train_data[0][ind] == "CapitalLoss"):
            removableFeatureIndex.append(ind)
    removableFeatureIndex.sort()
    for index in range(len(removableFeatureIndex)):
        removableFeatureIndex[index] = removableFeatureIndex[index] - index

    for indexForRenovableFeatures in removableFeatureIndex:
        # Unnecessary features as mentioned in the question
        for trainIng_data_instance in train_data:
            del trainIng_data_instance[indexForRenovableFeatures]
        for test_data_instance in test_data:
            del test_data_instance[indexForRenovableFeatures]
        # Test data can not contain class label so removing class label from test_data

        # if(train_data[0][ind] == "Income"):
    for test_data_instance in test_data:
        del test_data_instance[len(train_data[0]) - 1]

# def convert(dataset, is_train_data):
#     # Must include header in this dataset 2d-list
#     if(is_train_data):
#         dataset = z_score(dataset, is_train_data)
#         dataset = one_hot_encoding(dataset, is_train_data)
#     return dataset

def convert(dataset, training_headers):
    # Must include header in this dataset 2d-list
    new_cols_to_add_in_train_test = {}
    return_data = []
    if(not training_headers):
        dataset = z_score(dataset)
        dataset = one_hot_encoding(dataset)
        return_data = copy.deepcopy(dataset)
    else:
        new_dataset = z_score_test(dataset, training_headers)
        (new_dataset, new_cols_to_add_in_train_test) = one_hot_encoding_test(dataset, new_dataset)
        return_data = copy.deepcopy(new_dataset)
    return (return_data, new_cols_to_add_in_train_test)

def z_score(dataset):
    columNummbersForZScore = []
    for x in range(len(dataset[0])):
        flag = 0
        # Here, this condition evaluates that we will not going to put Z-score on class labels (assuming last column as class label)
        if(x == len(dataset[0]) - 1):
            continue
        for y in range(len(dataset)):
            if(y == 0):
                continue
            if(type(dataset[y][x]) == str):
                flag = 1
                break
        if(not flag):
            columNummbersForZScore.append(x)
    for featureIndex in columNummbersForZScore:
        sum = 0
        for point in range(len(dataset)):
            if(point == 0):
                continue
            sum = sum + dataset[point][featureIndex]
        mean = float(sum/(len(dataset)-1))
        squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation = 0
        for point in range(len(dataset)):
            if(point == 0):
                continue
            squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation = squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation + (dataset[point][featureIndex] - mean)**2
        standard_deviation = float(float(squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation/(len(dataset) - 1))**0.5)
        for point in range(len(dataset)):
            if(point == 0):
                continue
            dataset[point][featureIndex] = float((dataset[point][featureIndex] - mean)/standard_deviation)
    return dataset

def one_hot_encoding(dataset):
    columNummbersForOneHotEncoding = []
    for x in range(len(dataset[0])):
        flag = 0
        # Here, this condition evaluates that we will not going to put Z-score on class labels (assuming last column as class label)
        if (x == len(dataset[0]) - 1):
            continue
        for y in range(len(dataset)):
            if(y == 0):
                continue
            if(type(dataset[y][x]) != str):
                flag = 1
                break
        if(not flag):
            columNummbersForOneHotEncoding.append(x)
    for ind in range(len(columNummbersForOneHotEncoding)):
        uniqueFeatureValues = {}
        for point in range(len(dataset)):
            if(point == 0):
                continue
            if(dataset[point][columNummbersForOneHotEncoding[ind]] in uniqueFeatureValues):
                continue
            else:
                uniqueFeatureValues[dataset[point][columNummbersForOneHotEncoding[ind]]] = 1
        for key in uniqueFeatureValues:
            for point in range(len(dataset)):
                if(point == 0):
                    dataset[point].insert(0, str(dataset[point][columNummbersForOneHotEncoding[ind]]) + '_' + str(key))
                else:
                    # dataset[point][0] = 1 if dataset[point][columNummbersForOneHotEncoding[ind] + 1] == key else 0
                    dataset[point].insert(0, 1 if dataset[point][columNummbersForOneHotEncoding[ind]] == key else 0)
            for i in range(ind, len(columNummbersForOneHotEncoding)):
                columNummbersForOneHotEncoding[i] = columNummbersForOneHotEncoding[i] + 1
        for point in range(len(dataset)):
            del dataset[point][columNummbersForOneHotEncoding[ind]]
        for i in range(ind, len(columNummbersForOneHotEncoding)):
            columNummbersForOneHotEncoding[i] = columNummbersForOneHotEncoding[i] - 1
    return dataset

def z_score_test(dataset, training_headers):
    columNummbersForZScore = []
    rows, cols = (len(dataset), len(training_headers))
    # new_dataset = [[0]*cols]*rows
    new_dataset = []
    for i in range(rows):
        new_dataset.append([])
        for j in range(cols):
            new_dataset[i].append(0)

    new_dataset[0] = training_headers

    for x in range(len(dataset[0])):
        flag = 0
        for y in range(len(dataset)):
            if (y == 0):
                continue
            if (type(dataset[y][x]) == str):
                flag = 1
                break
        if (not flag):
            columNummbersForZScore.append(x)
    newColumNummbersForZScore = {}
    for value in columNummbersForZScore:
        newColumNummbersForZScore[value] = new_dataset[0].index(dataset[0][value])
    # for col in range(len(dataset[0])):
    #     for point in range(len(dataset)):
    #         new_col_ind = new_dataset[0].index()
    for featureIndex in columNummbersForZScore:
        sum = 0
        for point in range(len(dataset)):
            if (point == 0):
                continue
            sum = sum + dataset[point][featureIndex]
        mean = float(sum / (len(dataset) - 1))
        squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation = 0
        for point in range(len(dataset)):
            if (point == 0):
                continue
            squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation = squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation + (
                        dataset[point][featureIndex] - mean) ** 2
        standard_deviation = float(float(squaredSumOfDifferenceBetweenPointAndMeanForStandardDeviation / (len(dataset) - 1)) ** 0.5)
        for point in range(len(dataset)):
            if (point == 0):
                continue
            new_dataset[point][newColumNummbersForZScore[featureIndex]] = float((dataset[point][featureIndex] - mean) / standard_deviation)
            # dataset[point][featureIndex] = float((dataset[point][featureIndex] - mean) / standard_deviation)
    return new_dataset

def one_hot_encoding_test(dataset, new_dataset):
    columNummbersForOneHotEncoding = []
    for x in range(len(dataset[0])):
        flag = 0
        # Here, this condition evaluates that we will not going to put Z-score on class labels (assuming last column as class label)
        if (x == len(dataset[0]) - 1):
            continue
        for y in range(len(dataset)):
            if (y == 0):
                continue
            if (type(dataset[y][x]) != str):
                flag = 1
                break
        if (not flag):
            columNummbersForOneHotEncoding.append(x)
    newColumNummbersForOneHotEncoding = {}
    for value in columNummbersForOneHotEncoding:
        newColumNummbersForOneHotEncoding[value] = [i for i in range(len(new_dataset[0])) if dataset[0][value] in new_dataset[0][i]]
    new_columns_in_test_data = {}
    for ind in range(len(columNummbersForOneHotEncoding)):
        for point_value in range(len(dataset)):
            if point_value == 0:
                continue
            flag = 0
            for val in newColumNummbersForOneHotEncoding[columNummbersForOneHotEncoding[ind]]:
                if((str(dataset[0][columNummbersForOneHotEncoding[ind]]) + '_' + dataset[point_value][columNummbersForOneHotEncoding[ind]]) == new_dataset[0][val]):
                    flag = 1
                    new_dataset[point_value][val] = 1
            if(not flag):
                if((str(dataset[0][columNummbersForOneHotEncoding[ind]]) + '_' + dataset[point_value][columNummbersForOneHotEncoding[ind]]) in new_columns_in_test_data):
                    new_columns_in_test_data[(str(dataset[0][columNummbersForOneHotEncoding[ind]]) + '_' + dataset[point_value][columNummbersForOneHotEncoding[ind]])].append(point_value)
                else:
                    new_columns_in_test_data[(
                                str(dataset[0][columNummbersForOneHotEncoding[ind]]) + '_' + dataset[point_value][
                            columNummbersForOneHotEncoding[ind]])] = [point_value]
    return (new_dataset, new_columns_in_test_data)

def addNewColsInConvertedData(new_cols_to_add, converted_train_data, converted_test_data):
    for key in new_cols_to_add:
        for ind in range(len(converted_train_data)):
            if(ind == 0):
                converted_train_data[ind].insert(0, key)
            else:
                converted_train_data[ind].insert(0,0)
        for ind in range(len(converted_test_data)):
            if (ind == 0):
                converted_test_data[ind].insert(0, key)
            else:
                converted_test_data[ind].insert(0, 0)
        for val in new_cols_to_add[key]:
            converted_test_data[val][0] = 1
    return (converted_train_data, converted_test_data)

def centroid(train_data):
    centroids = initCentroids(train_data)
    K = len(centroids)
    print('This is K for Centroid method: ', K)

    stop = 0
    running_sts = 0
    running_sts_max = 100
    last_assigned_centroids = {}
    while (not stop):

        # stop the loop after some iteration
        if running_sts > running_sts_max:
            break

        # copy centroids to last_assigned_centroids to check whether centroids are getting changed or not in each iteration
        # if it is stable then break the loop.
        for centroid in centroids:
            last_assigned_centroids[centroid] = centroids[centroid]

        assigned_clusters = {}

        for data in train_data:
            min_dist = math.inf
            for centroid in centroids:
                # Calculate the distance between each centroid and data point
                dist_to_centroid = distance(centroids[centroid], data[:(len(data) - 1)])
                # update minimum distance and cluster number(class lable)
                if dist_to_centroid < min_dist:
                    min_dist = dist_to_centroid
                    assigned_cluster_class = centroid
            if (assigned_cluster_class in assigned_clusters):
                assigned_clusters[assigned_cluster_class].append(data[:(len(data) - 1)])
            else:
                assigned_clusters[assigned_cluster_class] = [data[:(len(data) - 1)]]

        # Update centroids by taking average of all instances which are assigned to same class from above looping code.
        for centroid in centroids:
            #             centroids[centroid] = dataset[np.array(assigned_clusters) == centroid].mean(axis=0)
            if (centroid in assigned_clusters and len(assigned_clusters[centroid]) > 1):
                # Here, We are computing sum for each feature, for example:
                # let's say we have 2 vectors assigned to one centroid: (1,2,3) and (4,5,6)
                # then we are doing (1+4, 2+5, 3+6) in the below line.
                centroids[centroid] = [sum(i) for i in zip(*assigned_clusters[centroid])]
                # Here, we are dividing the resultant sum with length of the assigned centroid list to get the mean for that centroid class.
                centroids[centroid] = [float(i / len(assigned_clusters[centroid])) for i in centroids[centroid]]

        # Append new centroids to an array for checking centroid changes purpose.
        #         original = []
        #         for centroid in centroids:
        #             original.append(centroids[centroid])

        # Check if last centroid and new centroid is same then centroids are stable and we do not need to iterate this again and again so stop the loop
        onTheSamePointAfterUpdatingCentroidsCount = 0
        for centroid in centroids:
            if (float(distance(last_assigned_centroids[centroid], centroids[centroid])) == 0.0):
                onTheSamePointAfterUpdatingCentroidsCount = onTheSamePointAfterUpdatingCentroidsCount + 1

        #         error = sum(sum([(x - y) ** 2 for x, y in zip(last_assigned_centroids, original[0])])) ** 0.5
        if onTheSamePointAfterUpdatingCentroidsCount == len(centroids):
            stop = 1

        running_sts += 1
        # print(running_sts)
    return centroids


def initCentroids(data_points):
    centroids = {}
    # Below loop will pick one lable and it's corresponding feature vector from the dataset
    for instance in data_points:
        # Assuming that our class label is th last column of the dataset, so we were taking first unique feature vecctor(value in dict) for the unique class(key in dict) and
        if (instance[len(instance) - 1] in centroids):
            continue
        else:
            centroids[instance[len(instance) - 1]] = instance[:(len(instance)-1)]
    return centroids

# This function is used to predict centroid classifier model which is written from scratch
def centroid_pred_updated(centroids, test_data):
    min_dist = math.inf
    for centroid in centroids:
        # Calculate the distance between each centroid and data point
        dist_to_centroid = distance(centroids[centroid], test_data)
#         dist_to_centroid = (sum([(x-y)**2 for x,y in zip(test_data,centroids[centroid])]))**0.5
        # update minimum distance and cluster number(class lable)
        if dist_to_centroid < min_dist:
            min_dist = dist_to_centroid
            assigned_cluster_class = centroid
    # Return the predicted value
    return assigned_cluster_class

def SVM(train_df, trainY, test_data, test_label, kernel):
    x = train_df
    y = trainY
    clf = SVC(kernel=kernel, gamma='auto')
    # Below line trains the model
    clf.fit(x, y)
    predictions = clf.predict(test_data)
    count = 0
    for actual_label, predicted_label in zip(test_label, predictions):
        if(actual_label == predicted_label):
            count = count + 1
    if(kernel == 'linear'):
        # This is w (weights):
        w = clf.coef_
        print("Weight: ", w)
        # This is margin:
        margin = float(1 / (sum(map(lambda i: i * i, w[0]))) ** 0.5)
        print("Margin: ", margin)
    # This is b (bias):
    b = clf.intercept_
    print("Bias: ", b)
    # This is Alpha :
    alpha = abs(clf.dual_coef_)
    print("Alpha: ", alpha)
    # return predicted test data lable for SVM
    return count

def run_on_knn(train_data, test_data, training_set):
    K = input('Please provide value for K to run KNN on the given data(default K = 5)')
    try:
        K = int(K)
    except:
        K = 5
    test_data_knn = copy.deepcopy(test_data)
    for test_d in test_data_knn:
        test_d.append(KNN(train_data[1:901], test_d, K))
        # Below loop is removing calculated pushed distance in each training_data from test_data_knn instance
        for trainIng_data_instance in train_data[1:901]:
            trainIng_data_instance.pop()
    count = 0
    for t in range(len(test_data_knn)):
        if (test_data_knn[t][len(test_data_knn[t]) - 1] == training_set[t + 901][len(training_set[0]) - 1]):
            count = count + 1
    final_accuracy=(count/len(test_data_knn))*100
    print("KNN Accuracy: ", str(final_accuracy))

def run_on_centroid(train_data, test_data, training_set):
    train_data_centroid = copy.deepcopy(train_data)
    (converted_train_data, new_cols_to_add) = convert(train_data_centroid, None)
    test_data_centroid = copy.deepcopy(test_data)
    test_data_centroid.insert(0, train_data[0][0:len(train_data[0]) - 1])
    (converted_test_data, new_cols_to_add) = convert(test_data_centroid,
                                                     train_data_centroid[0][0:len(train_data_centroid[0]) - 1])
    if (len(new_cols_to_add) > 0):
        (converted_train_data, converted_test_data) = addNewColsInConvertedData(new_cols_to_add, converted_train_data,
                                                                                converted_test_data)
    trained_centroids = centroid(converted_train_data[1:])
    for ind in range(len(converted_test_data)):
        if (ind == 0):
            converted_test_data[ind].append('Income')
        else:
            converted_test_data[ind].append(centroid_pred_updated(trained_centroids, converted_test_data[ind]))
    # print('Test')
    count = 0
    for t in range(0, 101):
        if (t == 0):
            continue
        if (converted_test_data[t][len(converted_test_data[t]) - 1] == training_set[t + 900][len(training_set[0]) - 1]):
            count = count + 1
        # else:
        #     print("Miss Match Centroid: ",t,converted_test_data[t][len(converted_test_data[t]) - 1])

    # Manual test Convert Methods.
    # (converted_train, new_cols_to_add) = convert([['age', 'type', 'won', 'class'],[11, "warm-blooded", 2, 1],[13, "cold-blooded", 3, 3]], None)
    # (converted_test, new_cols_to_add) = convert([['age', 'type', 'won'] , [12, "warm-blooded", 4],[15, "cold-x-blooded", 10]], converted_train[0][0:4])
    # if (len(new_cols_to_add) > 0):
    #     (converted_train, converted_test) = addNewColsInConvertedData(new_cols_to_add, converted_train, converted_test)

    print('Centroid Accuracy: ', str(count))

def run_on_SVM(train_data, test_data, training_set):
    train_data_svm = copy.deepcopy(train_data)
    test_data_svm = copy.deepcopy(test_data)

    (converted_train_data_svm, new_cols_to_add_svm) = convert(train_data_svm, None)
    test_data_svm.insert(0, train_data[0][0:len(train_data[0]) - 1])
    (converted_test_data_svm, new_cols_to_add_svm) = convert(test_data_svm,
                                                             train_data_svm[0][0:len(train_data_svm[0]) - 1])
    if (len(new_cols_to_add_svm) > 0):
        (converted_train_data_svm, converted_test_data_svm) = addNewColsInConvertedData(new_cols_to_add_svm,
                                                                                        converted_train_data_svm,
                                                                                        converted_test_data_svm)

    train_class = []
    for instance in converted_train_data_svm:
        train_class.append(instance.pop())
    test_pred_linear = []
    test_pred_gaussian = []
    actual_test_labels = []
    for t in range(0, 100):
        actual_test_labels.append(training_set[t + 901][len(training_set[0]) - 1])
    true_pred_linear_svm = SVM(converted_train_data_svm[1:], train_class[1:], converted_test_data_svm[1:],
                               actual_test_labels, 'linear')
    true_pred_gaussian_svm = SVM(converted_train_data_svm[1:], train_class[1:], converted_test_data_svm[1:],
                                 actual_test_labels, 'rbf')

    print('SVM linear Accuracy: ', float(true_pred_linear_svm * 100 / len(actual_test_labels)))
    print('SVM gaussian Accuracy: ', float(true_pred_gaussian_svm * 100 / len(actual_test_labels)))

def Main():
    training_set = loadDataset('./american_people_1000.txt')
    train_data = copy.deepcopy(training_set[:901])
    test_data = copy.deepcopy(training_set[901:])

    dropUnnecessaryFeatures(train_data, test_data)
    # KNN Implementation Call
    run_on_knn(train_data, test_data, training_set)
    # Centroid Implementation Call
    run_on_centroid(train_data, test_data, training_set)
    # SVM Implementation Call
    run_on_SVM(train_data, test_data, training_set)

if __name__ == '__main__':
    Main()