import tensorflow as tf
import timeit
from datetime import datetime

print('\n', '그래디언트 테이프 예제')
x = tf.ones((2, 2))
# -> [[1, 1],
#     [1, 1]]

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# 입력 텐서 x에 대한 z의 도함수
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0

# # 테이프 사용하여 중간값 y에 대한 도함수를 계산합니다. 
# dz_dy = t.gradient(z, y)
# assert dz_dy.numpy() == 8.0

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # 테이프에 대한 참조를 삭제합니다.


print('\n', '제어 흐름 기록 예제')
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0 #-> assert 뒤의 조건이 True가 아니면 AssertError 를 발생
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


print('\n', '고계도 그래디언트 예제')
x = tf.Variable(1.0)  # 1.0으로 초기화된 텐서플로 변수를 생성합니다.

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # 't' 컨텍스트 매니저 안의 그래디언트를 계산합니다.
  # 이것은 또한 그래디언트 연산 자체도 미분가능하다는 것을 의미합니다. 
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0


print('\n', '그래프 추적하는 예제')
# Define a Python function
def function_to_get_faster(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Create a `Function` object that contains a graph
a_function_that_uses_a_graph = tf.function(function_to_get_faster)

# Make some tensors
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

# It just works!
a_function_that_uses_a_graph(x1, y1, b1).numpy()
print(a_function_that_uses_a_graph(x1, y1, b1).numpy())
# -> tf.function a_function_that_uses_a_graph 안에 function_to_get_faster 가 들어 있음.

def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Use the decorator
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes inner_function() as well as outer_function()
outer_function(tf.constant([[1.0, 2.0]])).numpy()
print(outer_function(tf.constant([[1.0, 2.0]])).numpy())


print('\n', '흐름 제어의 부작용')
def my_function(x):
  if tf.reduce_sum(x) <= 1:
    return x * x
  else:
    return x-1

a_function = tf.function(my_function)

print("First branch, with graph:", a_function(tf.constant(1.0)).numpy())
print("Second branch, with graph:", a_function(tf.constant([5.0, 5.0])).numpy())

# Don't read the output too carefully.
print(tf.autograph.to_code(my_function))


print('\n', '속도 향상 예제')
# Create an oveerride model to classify pictures
class SequentialModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return x

input_data = tf.random.uniform([60, 28, 28])

eager_model = SequentialModel()
graph_model = tf.function(eager_model)

print("Eager time:", timeit.timeit(lambda: eager_model(input_data), number=10000))
print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=10000))
# -> tf.function을 사용한 경우가 eager mode 에 비해 속도가 빠르다.


print('\n', '다형 함수 예제')
print(a_function)
# a_function = tf.function(my_function)

print("Calling a `Function`:")
print("Int:", a_function(tf.constant(2)))
print("Float:", a_function(tf.constant(2.0)))
print("Rank-1 tensor of floats", a_function(tf.constant([2.0, 2.0, 2.0])))

# Get the concrete function that works on floats
# -> get_concrete_function 은 tf.function의 메서드
print("Inspecting concrete functions")
print("Concrete function for float:")
print(a_function.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.float32)))
print("Concrete function for tensor of floats:")
print(a_function.get_concrete_function(tf.constant([2.0, 2.0, 2.0])))

# Concrete functions are callable
# Note: You won't normally do this, but instead just call the containing `Function`
cf = a_function.get_concrete_function(tf.constant(2))
print("Directly calling a concrete function:", cf(tf.constant(2)))


print('\n', 'run_eagerly=True 사용 예제')
# Define an identity layer with an eager side effect
class EagerLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(EagerLayer, self).__init__(**kwargs)
    # Do some kind of initialization here

  def call(self, inputs):
    print("\nCurrently running eagerly", str(datetime.now()))
    return inputs

# Create an override model to classify pictures, adding the custom layer
class SequentialModel_1(tf.keras.Model):
  def __init__(self):
    super(SequentialModel_1, self).__init__()
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)
    self.eager = EagerLayer()

  def call(self, x):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return self.eager(x)

# Create an instance of this model
model = SequentialModel_1()

# Generate some nonsense pictures and labels
input_data = tf.random.uniform([60, 28, 28])
labels = tf.random.uniform([60])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# from_logits 란??
# from_logits=False : 클래스 분류 문제에서 softmax 함수를 거칠때(default) 
# from_logits=True : 그렇지 않을 때
# 딥러닝에서의 logit -> 모델의 출력값이 문제에 맞게 normalize되었느냐
# sigmoid 또는 linear를 거쳐서 만들어 지면 True
# softmax 일때, logit -> false

model.compile(run_eagerly=False, loss=loss_fn)

model.fit(input_data, labels, epochs=3)

print("Running eagerly")
# When compiling the model, set it to run eagerly
model.compile(run_eagerly=True, loss=loss_fn)

model.fit(input_data, labels, epochs=1)


print('\n', 'run_functions_eagerly 사용 예제')
# Now, globally set everything to run eagerly
# tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)

print("Run all functions eagerly.")

# First, trace the model, triggering the side effect
polymorphic_function = tf.function(model)

# It was traced...
print(polymorphic_function.get_concrete_function(input_data))

# But when you run the function again, the side effect happens (both times).
result = polymorphic_function(input_data)
result = polymorphic_function(input_data)

# Don't forget to set it back when you are done
# tf.config.experimental_run_functions_eagerly(False)
tf.config.run_functions_eagerly(False)


print('\n', "추적 및 성능 예제")
# Use @tf.function decorator
@tf.function
def a_function_with_python_side_effect(x):
  print("Tracing!")  # This eager
  return x * x + tf.constant(2)

# This is traced the first time
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect
print(a_function_with_python_side_effect(tf.constant(3)))

# This retraces each time the Python argument chances
# as a Python argument could be an epoch count or other
# hyperparameter
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))
# result -> 
# Tracing!
# tf.Tensor(6, shape=(), dtype=int32)
# tf.Tensor(11, shape=(), dtype=int32)
# Tracing!
# tf.Tensor(6, shape=(), dtype=int32)
# Tracing!
# tf.Tensor(11, shape=(), dtype=int32)

