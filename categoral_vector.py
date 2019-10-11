measure = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.}
]

measure_x = [
    {'city': 10, 'temperature': 33.},
    {'city': 12, 'temperature': 12.},
    {'city': 20, 'temperature': 18.}
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(measure)

#
# array([[ 1,  0,  0, 33],
#       [ 0,  1,  0, 12],
#       [ 0,  0,  1, 18]], dtype=int32)
       
print(vec.get_feature_names())
# ['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=True, dtype=int)
X = vec.fit_transform(measure)
# <3x4 sparse matrix of type '<class 'numpy.int32'>'
# with 6 stored elements in Compressed Sparse Row format>
print(X)

X.toarray()
# array([[ 1,  0,  0, 33],
#       [ 0,  1,  0, 12],
#       [ 0,  0,  1, 18]], dtype=int32)