#################################
#           메  모              #
#################################


runtime and Unified functions
eager execution

@tf.function 안의 코드도 이 효과로 쓰여진 순서대로 실행

In tf 1.x was used sesstion.run(), but in 2.0, tf.function() will be used.


"""차이점"""
# 텐서플로 1.x
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# 텐서플로 2.0
outputs = f(input)

텐서플로 1.x의 일반적인 사용 패턴은 키친 싱크 전략 입니다. 먼저 모든 연산을 결합하여 준비한 다음 session.run()을 사용해 선택한 텐서를 평가합니다.

텐서플로 2.0에서는 필요할 때 호출할 수 있는 작은 함수로 코드를 리팩토링 해야 합니다. 모델 훈련의 한단계나 정방향 연산 같은 고수준 연산에만 tf.function 데코레이터를 적용하세요.


기본 버전
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

# 여전히 w_i, b_i 변수를 직접 관리해야 합니다. 이 코드와 떨어져서 크기가 정의됩니다.


케라스 버전
# 각 층은 linear(x)처럼 호출 가능합니다.
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]


## example ##
전이 학습을 통한 예제
몸통(trunk)을 공유하는 다중 출력 모델 훈련

trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# 주된 데이터셋에서 훈련합니다.
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    prediction = path1(x) -> x is a list of dataset in python.
    loss = loss_fn_head1(prediction, y)

  # trunk와 head1 가중치를 동시에 최적화합니다. ( <- 요 다음 과정에 대한 설명)
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# trunk를 재사용하여 head2를 세부 튜닝합니다.
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    prediction = path2(x) -> x는 리스트
    loss = loss_fn_head2(prediction, y)

  # trunk 가중치는 제외하고 head2 가중치만 최적화합니다.
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables)) -> head2의 가중치만 최적화, path2가 아니라

# trunk 연산만 재사용을 위해 저장할 수 있습니다.
tf.saved_model.save(trunk, output_path) -> 학습은 head와 path를 모두 하였지만 trunk만 저장

메모리 크기에 맞는 훈련 데이터를 반복할 때는 보통의 파이썬 반복자를 사용해도 좋습니다. 그렇지 않다면 디스크에서 훈련 데이터를 읽는 가장 좋은 방법은 tf.data.Dataset 입니다.

tf.function 은 오토그래프를 사용하여 파이썬 반복문을 동일한 그래프 연산으로 바꾸어 줍니다.

@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset: -> dataset은 zip 되어 있는 데이터
    with tf.GradientTape() as tape:
      prediction = model(x)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

케라스의 model.fit() API를 사용하면 데이터셋 반복에 관해 신경 쓸 필요가 없습니다.

model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)


오토그래프는 데이터에 따라 결정되는 제어흐름을 tf.cond 와 tf.while_loop 같은 그래프 모드 연산으로 변환 시켜 줍니다.

class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])

    outputs = tf.TensorArray(tf.float32, input_data.shape[0]) -> input_data.shape[0]는 size
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)

    for i in tf.range(input_data.shape[0]): -> time dependency
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)


    # [time, batch, features] -> [batch, time, features]
    return tf.transpose(outputs.stack(), [1, 0, 2]), state


tf.metrics과 tf.summary로 기록하기

## tf.summary 사용
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)

## tf.metircs 사용
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  loss = loss_fn(model(test_x), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)



##
##
## EAGER EXECUTION


텐서플로의 즉시 실행은 그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경입니다.
나중에 실행하기 위해 계산 가능한 그래프를 생성하는 대신에 계산값을 즉시 알려주는 연산입니다.

이제 부터는 텐서플로우 연산을 바로 실행할 수 있고 결과를 즉시 확인할 수 있습니다.

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

즉시 실행 활성화는 텐서플로우 연산을 바로 평가하고 그 결과를 파이썬에게 알려주는 방식으로 동작을 변경합니다.
-> print()나 디버거를 통해서 결과를 검토하기 쉽습니다.
텐서값을 평가, 출력 하거나 확인하는 것이 그래디언트를 계산하는 흐름을 방해하지 않습니다.

즉시 실행은 numpy와 같이 작동.
tf.Tensor.numpy 매서더는 객체 값을 numpy ndarray로 변환합니다.


## 동적 제어 흐름

즉시 실행의 가장 큰 이점은 모델을 실행하는 동안에도 호스트 언어의 모든 기능을 활용할 수 있다는 것입니다. 


## 즉시 훈련

# 그래디언트 계산하기

자동 미분은 인공 신경망 훈련을 위한 역전파와 같은 기계학습 알고리즘을 구현하는데 유용합니다.

** 즉시 실행을 하는 동안에는 나중에 그래디언트를 계산하는 연산을 추적하기 위해 tf.GradientTape를 사용하세요.
** 복잡하고 반복적인 훈련인 경우에 더 유용.
-> 매번 실행될 때 서로 다른 연산이 수행될 수 있기 때문에 모든 정방향 연산은 "tape"에 기록됩니다. 그 다음 tape를 거꾸로 돌려 그래디언트를 계산한 후 tape를 폐기합니다.

tf.GradientTape 는 오직 하나의 그래디언트만을 계산할 수 있고 부가적인 호출은 실행중 에러를 발생시킵니다.


## 모델 훈련 예제

MNIST 를 사용한 모델 훈련 예제



## 변수와 옵티마이저

tf.Variable 객체는 자동 미분을 쉽게 하기 위해서 학습동안 변경된 tf.Tensor 값을 저장합니다. 모델 파라미터는 클래스 인스턴스 변수로 캡슐화 될 수 있음.

효과적인 캡슐화를 위해 tf.Variable와 tf.GradientTape를 함께 사용


## 즉시 실행에서 상태를 위한 객체 사용

tf.Sesstion에 의해 관리 되던 tf 1.x 버전
반면, 즉시 실행 상태에서의 객체 수명은 그와 관련된 파이썬 객체 수명에 의해 결정
  -> tf.Session 종속이 아니라 파이썬 종속 

# 변수는 객체다.
즉시 실행에서 변수는 그 객체의 마지막 참조가 제거될 때까지 유지되고 그 이후 삭제됩니다.

# 객체 기반의 저장
tf.train.Checkpoint는 tf.Variable을 체크포인트 파일로 저장하거나 체크포인트 파일에서 복수할 수 있습니다.

tf.train.Checkpoint는 숨겨진 변수를 요구하지 않고 객체 내부 상태를 저장합니다. 옵티마이저와 모델 전역 단계 상태를 기록하려면 tf.train.Checkpoint에 전달하면 됨.

# 객체 지향형 지표
tf.keras.metrics 는 객체로 저장. 새로운 데이터를 객체에 전달, 지표를 수정, tf.keras.metrics.result 메서드를 사용해 그 결과를 얻음.

# 서머리와 텐서보드
텐서보드는 훈련과정에서 모델을 파악하거나 디버깅하고 최적화하기 위해 사용하는 시각화 도구
텐서보드는 summary 이벤트를 사용
즉시 실행에서 변수의 summary정보를 기록하기 위해서 tf.summary를 사용



## 자동 미분 관련 고오급편

# 동적 모델
tf.GradientTape는 또한 동적인 모델에서도 사용가능

** 역추적 길찾기 알고리즘 예제는 그래디언트가 있으며 미분 가능하다는 것을 제외하면 일반적인 numpy로 작성한 코드처럼 보임.

# 사용자 정의 그래디언트
사용자 정의 그래디언트는 그래디언트를 재정의 하는 가장 쉬운 방법.
forward propagation 내에서 입력값 또는 출력값, 중간값과 관련된 그래디언트를 정의
예제는 역전파 과정에서 그래디언트의 norm을 clip하는 가장 쉬운 방법

사용자 정의 그래디언트는 일반적으로 연산에 대해 수치적으로 안정된 그래디언트를 제공하기 위해 사용됩니다.


예제에서 log1pexp 함수는 이론적으로 사용자 정의 그래디언트를 활용해 간결해 질 수 있습니다. 
예제는 불필요한 계산을 제거함으로써 계산을 좀 더 효율적으로 하기 위해 정방향 경로안에서 계산된 tf.exp(x) 값을 재사용합니다.



## 성능
즉시 실행에서 계산은 자동으로 GPU로 분배됨.
계산 분배를 사용자가 제어하고 싶다면 그 부분을 tf.device('/gpu:0') 블록으로 감싸서 실행하세요.

tf.Tensor 객체는 실제로 그 연산을 수행할 다른 디바이스로 복사될 수 있습니다.



### 텐서 소개

텐서는 일관된 유형(dtype)을 가진 다차원 배열
numpy에 익숙하다면 텐서는 일종의 np.arrays 와 같습니다.
모든 텐서는 파이썬 숫자 및 문자열과 같이 변경할 수 없습니다.
텐서의 내용을 업데이트할 수 없으며, 새로운 텐서를 만들 수만 있습니다.

## 기초





