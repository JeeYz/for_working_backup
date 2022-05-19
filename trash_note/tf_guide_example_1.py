import tensorflow as tf

# GPU 사용 가능 여부 판단
if tf.config.experimental.list_physical_devices("GPU"):
  with tf.device("gpu:0"): # 있다면,/ 존재한다면,
    print("GPU 사용 가능")
    v = tf.Variable(tf.random.normal([1000, 1000]))
    v = None  # v는 더이상 GPU 메모리를 사용하지 않음


# x에 tf.Variable 할당
x = tf.Variable(10.)
# x 를 checkpoint로 설정
checkpoint = tf.train.Checkpoint(x=x)

x.assign(2.)   # 변수에 새로운 값을 할당하고 저장
checkpoint_path = './ckpt/'
checkpoint.save('./ckpt/')

x.assign(11.)  # 저장한 후에 변수 변경

# 체크포인트로부터 값을 복구
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x)  # => 2.0
# 저장한 후에 변수 변경은 할 수 없음
print(x.numpy())


import os

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint_dir = 'path/to/model_dir'

if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# root 클래스에 optimizer와 model 에 대해 변수 설정
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model)

# 저장과 로드
root.save(checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))


# 객체 지향형 지표
m = tf.keras.metrics.Mean("loss")
print(m) # -> 데이터의 형태와 메모리 값을 반환
m(0)
m(5)
print(m.result().numpy()) # -> 0과 5의 평균
m([8, 9])
print(m.result().numpy()) # -> 0과 5와 8과 9의 평균


# 서머리와 텐서보드
logdir = "./tb/"
# 주어진 log directory에 대해 summary를 기록
writer = tf.summary.create_file_writer(logdir)

with writer.as_default():  # 또는 반복 전에 writer.set_as_default()를 호출
  for i in range(1000):
    step = i + 1
    # 실제 훈련 함수로 손실을 계산
    loss = 1 - 0.001 * step
    if step % 100 == 0:
      # scalar summary를 기록
      # writer 에 기록
      tf.summary.scalar('손실', loss, step=step)


# 동적 모델
# 역추적 길찾기 알고리즘

def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # 변수는 자동적으로 기록되지만 텐서는 사용자가 스스로 확인해야 함
    # tape 추적
    tape.watch(init_x)
    # fn이란 객체에 init_x 입력
    value = fn(init_x)

  grad = tape.gradient(value, init_x) # -> 그래디언트 연산
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value

  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value

# 사용자 정의 그래디언트
# @tf.custom_gradient 
# -> custom gradient로 함수를 정의 한다.
#
# ! ! ! 아래 함수가 뭘 하려는지 잘 모르겠음. ! ! !
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    print("*** {}".format(dresult))
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn

# 사용자 정의 그래디언트
def log1pexp(x):
  return tf.math.log(1 + tf.exp(x))

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    # watch 메서드는 테이프에 의해 추적 된다.
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

# 그래디언트 계산은 x = 0일 때 잘 동작
print(grad_log1pexp(tf.constant(0.)).numpy()) # -> 0.5

# 그러나, x = 100일 때 수치적으로 불안정하기 때문에 실패
print(grad_log1pexp(tf.constant(100.)).numpy()) # -> nan


# 사용자 정의 그래디언트 예제
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    print("*** {}".format(dy))
    # 이 부분은 수학적으로 이해해야 함.
    return dy * (1 - 1 / (1 + e)) # -> sigmoid
  return tf.math.log(1 + e), grad # -> @tf.custom_gradient 의 Bind 메서드

#
# dy 의 정의
# With this definition, the gradient at x=100 will be correctly evaluated as 1.0.
# The variable dy is defined as the upstream gradient. i.e. the gradient from all the layers or functions originating from this layer.
# By chain rule we know that dy/dx = dy/x_0 * dx_0/dx_1 * ... * dx_i/dx_i+1 * ... * dx_n/dx
# In this case the gradient of our current function defined as dx_i/dx_i+1 = (1 - 1 / (1 + e)). The upstream gradient dy would be dx_i+1/dx_i+2 * dx_i+2/dx_i+3 * ... * dx_n/dx. The upstream gradient multiplied by the current gradient is then passed downstream.
# In case the function takes multiple variables as input, the grad function must also return the same number of variables. We take the function z = x * y as an example.
#

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

# 전처럼, 그래디언트 계산은 x = 0일 때 잘 동작
print(grad_log1pexp(tf.constant(0.)).numpy())

# 그래디언트 계산은 x = 100일 때 역시 잘 동작
print(grad_log1pexp(tf.constant(100.)).numpy())


print('\n')
print("성능 예제")
# 성능 예제
import time

def measure(x, steps):
  # 텐서플로는 처음 사용할 때 GPU를 초기화, 시간계산에서 제외
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul는 행렬 곱셈을 완료하기 전에 결과를 반환할 수 있습니다
  # (예, CUDA 스트림 대기열에 연산을 추가한 후에 결과를 반환할 수 있다).
  # 아래 x.numpy() 호출은 대기열에 추가된 모든 연산이 완료될 것임을 보장합니다
  # (그리고 그 결과가 호스트 메모리에 복사될 것이고,
  # 그래서 matnul 연산시간보다는 조금 많은 연산시간이
  # 포함됩니다).
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("{} 크기 행렬을 자기 자신과 {}번 곱했을 때 걸리는 시간:".format(shape, steps))

# CPU에서 실행:
with tf.device("/cpu:0"):
  print("CPU: {} 초".format(measure(tf.random.normal(shape), steps)))

# GPU에서 실행, 가능하다면:
if tf.config.experimental.list_physical_devices("GPU"):
  with tf.device("/gpu:0"):
    print("GPU: {} 초".format(measure(tf.random.normal(shape), steps)))
else:
  print("GPU: 없음")



if tf.config.experimental.list_physical_devices("GPU"):
  x = tf.random.normal([10, 10])

  x_gpu0 = x.gpu()
  x_cpu = x.cpu()

  _ = tf.matmul(x_cpu, x_cpu)    # CPU에서 실행
  _ = tf.matmul(x_gpu0, x_gpu0)  # GPU:0에서 실행




