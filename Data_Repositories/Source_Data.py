
def get_base_data():

    from sklearn.datasets import fetch_openml

    mnist = fetch_openml(name="mnist_784")

    return mnist
