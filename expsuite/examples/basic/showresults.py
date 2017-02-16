from suite import MySuite

# create the object of your suite
mysuite = MySuite()

# list of all experiments with get_exps()
exps = mysuite.get_exps()
print "list of available experiments:", exps


print "last value of 'beta' in repetition number 2 (2.log):", \
    mysuite.get_value(exps[0], 2, 'beta', 'last')

print "biggest value of 'iter' in repetition number 0 (0.log):", \
    mysuite.get_value(exps[0], 0, 'iter', 'max')

print "history of values of 'iter' in repetition number 4 (4.log):", \
    mysuite.get_history(exps[0], 4, 'iter')

