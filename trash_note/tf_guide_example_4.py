import tensorflow as tf
from datetime import datetime

# %load_ext tensorboard

class SimpleModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    self.a_variable = tf.Variable(5.0, name="train_me")
    self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
  def __call__(self, x):
    return self.a_variable * x + self.non_trainable_variable

simple_module = SimpleModule(name="simple")

simple_module(tf.constant(5.0))
print(simple_module(tf.constant(5.0)))


# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)


print('\n', '밀집 선형 레이어 예제')
class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

class SequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

print("Submodules:", my_model.submodules)

for var in my_model.variables:
  print(var, "\n")


print('\n', '변수 생성 연기하기 예제')
class FlexibleDenseModule(tf.Module):
  # Note: No need for `in+features`
  def __init__(self, out_features, name=None):
    super().__init__(name=name)
    self.is_built = False
    self.out_features = out_features

  def __call__(self, x):
    # Create variables on first call.
    if not self.is_built:
      self.w = tf.Variable(
        tf.random.normal([x.shape[-1], self.out_features]), name='w')
      self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
      self.is_built = True

    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

# Used in a module
class MySequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = FlexibleDenseModule(out_features=3)
    self.dense_2 = FlexibleDenseModule(out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

my_model = MySequentialModule(name="the_model")
# -> 요 밑에 것을 실행할 때 변수 생성됨.
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))


print('\n', '가중치 저장하기 예제')
chkp_path = "my_checkpoint"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)
checkpoint.write(chkp_path)

tf.train.list_variables(chkp_path)
print(tf.train.list_variables(chkp_path))

# 아래 코드를 통해 new_model과 new_checkpoint 가 연결됨
new_model = MySequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore("my_checkpoint")

# Should be the same result as above
new_model(tf.constant([[2.0, 2.0, 2.0]]))
print(new_model(tf.constant([[2.0, 2.0, 2.0]])))


print('\n', '함수 저장 예제')
class MySequentialModule_1(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    # super 함수 사용법
    # python 2.x 문법
    # super(ClassName, self).__init__()
    # super(ClassName, self).__init__(**kwargs)

    # python 3.x 문법
    # super().__init__()
    # super().__init__(**kwargs)

    # 범용성을 위해서는 python 2.x 방법으로 사용

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  @tf.function
  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model with a graph!
# 이 모듈은 이전과 똑같이 동작
my_model = MySequentialModule_1(name="the_model_1")

print(my_model([[2.0, 2.0, 2.0]]))
print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))

print('\n', '텐서 보드 코드')
# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)

# Create a new model to get a fresh trace
# Otherwise the summary will not see the graph.
new_model = MySequentialModule_1()

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = print(new_model(tf.constant([[2.0, 2.0, 2.0]])))

with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)

print('\n', 'SavedModel 생성하기 예제')
tf.saved_model.save(my_model, "the_saved_model")

# 새 객체로 로드
new_model = tf.saved_model.load("the_saved_model")
# new_model 은 MySequentialModule_1 과는 다른 객체
print(isinstance(new_model, MySequentialModule_1))

print(my_model([[2.0, 2.0, 2.0]]))
print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))


print('\n', 'Keras 레이어 예제')
class MyDense(tf.keras.layers.Layer): # -> Layer 클래스 상속
  # Adding **kwargs to support base Keras layer arguemnts
  def __init__(self, in_features, out_features, **kwargs):
    super().__init__(**kwargs) # -> python 3.x 방식

    # This will soon move to the build step; see below
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def call(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

simple_layer = MyDense(name="simple", in_features=3, out_features=3)

simple_layer([[2.0, 2.0, 2.0]])
print(simple_layer([[2.0, 2.0, 2.0]]))

print('\n', 'build 단계 예제')
class FlexibleDense(tf.keras.layers.Layer):
  # Note the added `**kwargs`, as Keras supports many arguments
  def __init__(self, out_features, **kwargs):
    super().__init__(**kwargs)
    self.out_features = out_features

  def build(self, input_shape):  # Create the state of the layer (weights)
    self.w = tf.Variable(
      tf.random.normal([input_shape[-1], self.out_features]), name='w')
    self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.matmul(inputs, self.w) + self.b

# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)

# 이 시점에서는 모델이 빌드되지 않았으므로 변수가 없음.
print(flexible_dense.variables)

# Call it, with predictably random results
print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))

print(flexible_dense.variables)

# build 는 한 번만 호출되므로 입력 shape가 레이어의 변수와 호환되지 않으면 입력이 거부됨.
try:
  print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
except tf.errors.InvalidArgumentError as e:
  print("Failed:", e)


print('\n', 'Keras 모델 예제')
class MySequentialModel(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.dense_1 = FlexibleDense(out_features=3)
    self.dense_2 = FlexibleDense(out_features=2)
  def call(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a Keras model!
my_sequential_model = MySequentialModel(name="the_model")

# Call it on a tensor, with random results
print("Model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))  

# variables 변수를 가지고 내부 variable을 확인할 수 있음.
print(my_sequential_model.variables)
print(my_sequential_model.submodules)

inputs = tf.keras.Input(shape=[3,])

x = FlexibleDense(3)(inputs)
x = FlexibleDense(2)(x)

my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)

my_functional_model.summary()

print(my_functional_model(tf.constant([[2.0, 2.0, 2.0]])))


print('\n', 'Keras 모델 저장하기 예제')
my_sequential_model.save("exname_of_file")

# 다시 로드
reconstructed_model = tf.keras.models.load_model("exname_of_file")

print(reconstructed_model(tf.constant([[2.0, 2.0, 2.0]])))