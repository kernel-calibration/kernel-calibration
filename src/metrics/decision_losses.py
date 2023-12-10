# Decision-maker's loss function for predicting which side of the median
# the target variable y falls on (See Section 6.3 of paper for more details)
def loss_fn_1(a, y):
        return 1.0*(y.sign() == a)