# save arrays in csv file
import numpy
a = numpy.array([[1,2],[3,4],[5,6]])
numpy.savetxt("foo.csv", a, delimiter=',', header="A,B", comments="")

# using pandas and dataframe

import pandas as pd
import numpy as np

a = np.array([1.2,2,3,4])
b = np.array([5,6,7,8.5])

df = pd.DataFrame({"name1" : a, "name2" : b})
df.to_csv("submission2.csv", index=False)
