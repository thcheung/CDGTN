from .experiment import *

experiment = {
    "Twitter15": Twitter15,
    "Twitter16": Twitter16,
    "CR_Twitter": CR_Twitter,
    "pheme": pheme,
    "weibo": weibo
}

def get_experiment(x):
    return experiment[x]