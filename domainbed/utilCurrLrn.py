import math

def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value
    
def reg_scheduler(sched_func, valMin, valMax, step, stepThresh = None, r = None, Ccor = None, Cdiv = None):
    if sched_func == "LINEAR":
        return (step / stepThresh) * valMax
    elif sched_func == "BINARY":
        return valMax if step >= stepThresh else valMin
    elif sched_func == "EXPONENTIAL":
        return (1-math.exp(-r*step)) * valMax
    else: #sched_func == "NONE"
        return valMax