import minpy.numpy as np
import minpy.dispatch.policy as policy
import time

# mp.set_policy(policy.OnlyNumPyPolicy())

start = time.time()
np.array([100, 100]).asnumpy()
end = time.time()
print 'First call:', end - start

start = time.time()
np.array([100, 100]).asnumpy()
end = time.time()
print 'Second call:', end - start

start = time.time()
np.array([100, 100]).asnumpy()
end = time.time()
print 'Third call:', end - start

