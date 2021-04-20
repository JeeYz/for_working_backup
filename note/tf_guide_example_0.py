import tensorflow as tf

print(tf.executing_eagerly())

a = tf.constant([[1, 2], [3, 4]])
print(a)

# support broadcasting
b = tf.add(a, 1)
print(b)

# 연산자 오버로딩
print(a*b)

# numpy 로 tensor 연산하기
import numpy as np

c = np.multiply(a, b)
print(c)

print(a.numpy()) # a는 tensor, a의 값를 numpy()로 불러올 수 있음.


# fizzbuzz

def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)

        if int(num%3) == 0 and int(num%5) == 0:
            print('fizzbuzz')
        elif int(num%3) == 0:
            print('fizz')
        elif int(num%5) == 0:
            print('buzz')
        else:
            print(num.numpy())
        counter += 1

fizzbuzz(15)


# tf.GradientTape

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w*w

grad = tape.gradient(loss, w)
print(grad)


# MNIST를 사용한 모델 훈련 예제

# mnist 데이터 가져오기 및 포맷 맞추기

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
    tf.cast(mnist_labels, tf.int64))
) # -> 아마도 numpy 데이터로부터 tensor 데이터 셋을 불러옴.

# 배치 사이즈로 나누고 섞어줌
dataset = dataset.shuffle(1000).batch(32)


# 모델 생성
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                            input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3,3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

# 즉시 실행에서는 훈련을 하지 않아도 모델을 사용하고 결과를 점검할 수 있습니다.
# 현재 코딩 상태가 즉시 실행 상태임. 일종의 @tf.function 상태임
for images, labels in dataset.take(1):
    print('logit : ', mnist_model(images[0:1]).numpy()) 
    # -> numpy()로 결과 값 확인
    # -> dataset에서 하나의 값을 취해서 결과 확인

# 즉시 실행 적용
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []

# tf.GradientTape() 적용
def train_step(images, labels):
    with tf.GradientTape() as tape: # -> tape에 정방향 연산 기록
        # logits = odds + log 
        # odds는 사건 A가 발생할 확률과 발생하지 않을 확률의 비율
        logits = mnist_model(images, training=True)

        # 결과의 형태를 확인하기 위해서 단언문 추가
        tf.debugging.assert_equal(logits.shape, (32, 10))

        # labels, logits를 통해 loss value 계산
        loss_value = loss_object(labels, logits)

    # history에 저장
    loss_history.append(loss_value.numpy().mean())
    # back propagation 계산
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    # 계산된 back propagation 최적화
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


def train():
    for epoch in range(3):
        # (batch, (images, labels)) 가 데이터 형태
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
        print('에포크 {} 종료'.format(epoch+1))

train()


# 훈련 그래프 그리기
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')


# 변수와 옵티마이저
# tf.keras.Model 상속
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    # tf.Variable로 선언
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# 약 3 * x + 2개의 점으로 구성된 실험 데이터
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# 최적화할 손실함수
# mean square error
# loss 계산 함수
def loss_0(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

# tf.GtadientTape().gradient
# gradient(
#   target, sources, output_gradients=None,
#   unconnected_gradients=tf.UnconnectedGradients.NONE)
# target :	a list or nested structure of Tensors or Variables to be differentiated.
# sources :	a list or nested structure of Tensors or Variables. target will be differentiated against elements in sources.

def grad_0(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss_0(model, inputs, targets)
    # loss_value -> target, [model.W, model.B] -> sources
    # tape에 기록
  return tape.gradient(loss_value, [model.W, model.B])

# 정의:
# 1. 모델
# 2. 모델 파라미터에 대한 손실 함수의 미분
# 3. 미분에 기초한 변수 업데이트 전략
model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("초기 손실: {:.3f}".format(loss_0(model, training_inputs, training_outputs)))

# 반복 훈련
for i in range(300):
  # 이 부분은 tape에 기록하는 부분
  grads = grad_0(model, training_inputs, training_outputs)
  # 이 부분에서 업데이트 발생
  # 실질 훈련이 실행되는 곳
  optimizer.apply_gradients(zip(grads, [model.W, model.B]))
  if i % 20 == 0:
    # 이 부분은 loss를 통해 계산만 실행
    print("스텝 {:03d}에서 손실: {:.3f}".format(i, loss_0(model, training_inputs, training_outputs)))

# loss 값만 구함
print("최종 손실: {:.3f}".format(loss_0(model, training_inputs, training_outputs)))
# 업데이트 된 tf.Variable의 값 확인
# 최초 값 weight = 5. , bias = 10.
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))



