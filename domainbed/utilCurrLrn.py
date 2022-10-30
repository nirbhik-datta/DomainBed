import math

def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value
    
def reg_scheduler(sched_func, valMin, valMax, step, stepThresh=None, r=None, Ccor=None, Cdiv=None, invert=False):
    if sched_func == "LINEAR":
        val = (step / stepThresh) * valMax
    elif sched_func == "BINARY":
        val = valMax if step >= stepThresh else valMin
    elif sched_func == "EXPONENTIAL":
        val = (1-math.exp(-r*step)) * valMax
    else: #sched_func == "NONE"
        return valMax

    if invert:
        return 1 - val
    else:
        return val
