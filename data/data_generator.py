from data import dataset_iris, dataset_diabetes, dataset_mnist, dataset_cifar, dataset_fashion


def get_data(key, test_size_percent):
    print("loading ", key, " dataset")
    if key == "iris":
        return dataset_iris.get_data(test_size_percent)
    elif key == "diabetes":
        return dataset_diabetes.get_data(test_size_percent)
    elif key == "mnist":
        return dataset_mnist.get_data(test_size_percent)
    elif key == "fashion":
        return dataset_fashion.get_data(test_size_percent)
    elif key == "cifar":
        return dataset_cifar.get_data(test_size_percent)
