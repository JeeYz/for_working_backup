텐서플로우 가이드 정리
=====================


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



## 텐서 소개

텐서는 일관된 유형(dtype)을 가진 다차원 배열
numpy에 익숙하다면 텐서는 일종의 np.arrays 와 같습니다.
모든 텐서는 파이썬 숫자 및 문자열과 같이 변경할 수 없습니다.
텐서의 내용을 업데이트할 수 없으며, 새로운 텐서를 만들 수만 있습니다.

## 기초
예제는 스칼라 텐서 입니다. 스칼라에 axis는 없습니다.
벡터 텐서는 1축 입니다.
행렬 텐서는 2축 입니다.

np.array 또는 tensor.numpy 메서드를 사용하여 텐서를 numpy 배열로 변환할 수 있습니다.


## 형상 정보

shape(형상) : 텐서의 각 차원의 길이(요소의 수)
rank(순위) : 텐서 차원의 수. 스칼라는 0, 벡터는 1, 행렬은 2
axis or dimension(축 또는 차원) : 텐서의 특정 차원
size(크기) : 텐서의 총 항목 수, 곱 형상 벡터

rank4 axes -> 0 : batch / 1 : width / 2 : height / 3 : features


### 인덱싱

## 단일축 인덱싱
텐서플로우는 파이썬 인덱싱 규칙과 numpy 인덱싱 기본 규칙을 따른다.
콜론 : 은, 슬라이스 start:stop:step에 사용

스칼라를 사용하여 인덱싱하면 차원이 제거 됩니다. -> 예제
: 조각으로 인덱싱하면 차원이 유지.

## 다축 인덱싱

더 높은 순위의 텐서는 여러 인덱스를 전달하여 인덱싱
단일 축의 경우에서와 정확히 같은 단일 축 규칙이 각 축에 독립적으로 적용


## 형상 조작하기

텐서의 형상 바꾸기 -> 예제
텐서를 새로운 형상으로 바꿀 수 있음. 기본 데이터를 복제할 필요가 없음. -> 예제

데이터의 레이아웃은 메모리에서 유지되고 요청된 형상이 같은 데이터를 가리키는 새 텐서가 작성.
텐서플로우는 C스타일의 행중심 메모리 순서를 사용. 
가장 오른쪽에 있는 인덱스를 증가시키면 메모리의 단일 단계에 해당.

일반적으로, tf.reshape의 합리적인 용도는 인접한 축을 결합하거나 분할 하는 것. (expand dims 포함)
3x2x5 텐서의 경우, 슬라이스가 혼합되지 않으므로 (3x2)x5 또는 3x(2x5)로 재구성하는 것이 합리적.

tf.reshape 에서 축 교환이 작동하지 않으면, tf.transpose를 수행

완전히 지정되지 않은 형상에서 실행할 수 있음.
형상에 None이 포함되거나 형상이 None인 경우.


## DTypes에 대한 추가 정보

tf.Tensor의 데이터 유형을 검사하려면, Tensor.dtype속성 사용
파이썬 객체에서 tf.Tensor를 만들때 선택적으로 데이터 유형을 지정할 수 있음.
텐서플로우는 데이터를 디폴트 유형으로 선택.
정수를 int32, 소수점 숫자를 tf.float32로 변환
텐서플로우는 numpy가 배열로 변활할 때 사용하는 것과 같은 규칙을 사용.


# 브로드 캐스팅

브로드 캐스팅은 numpy의 해당 특성에서 빌린 개념.
특정 조건에서 작은 텐서는 연산을 실행할 때 더 큰 텐서에 맞게 자동으로 확장(streched) 됨.
스칼라에 텐서를 곱하거나 더할 때, 스칼라는 다른 인수와 같은 형상으로 브로드캐스트 됨.

크기가 1인 차원은 다른 인수와 일치하도록 확장될 수 있음.
두 인수 모두 같은 계산으로 확장될 수 있음.
3x1행렬에 요소별로 1x4행렬을 곱하여 3x4 행렬을 만듬. -> 예제

브로드 캐스팅은 브로드캐스트 연산으로 메모리에서 확장된 텐서를 구체화하지 않음
tf.broadcast_to 를 사용한 예제


## tf.convert_to_tensor

tf.matmul 및 tf.reshape와 같은 대부분의 ops는 클래스 tf.Tensor의 인수를 사용.
대부분의 ops는 텐서가 아닌 인수에 대해 convert_to_tensor를 호출
변환 레지스트리가 있음.
numpy의 ndarray, tensorshape, 파이썬 list 및 tf.Variable과 같은 대부분의 객체 클래스는 모두 자동으로 변환


## 비정형 텐서

어떤 축을 따라 다양한 수의 요소를 가진 텐서를 "비정형(ragged)"
비정형 데이터는 tf.ragged.RaggedTensor를 사용
비정형 텐서는 정규 텐서로 표현할 수 없음.


## 문자열 텐서

tf.string은 dtype이며, 텐서에서 문자열과 같은 데이터를 나타냄
문자열은 원자성이므로 파이썬 문자열과 같은 방식으로 인덱싱
문자열의 길이는 텐서의 차원이 될 수 없음.

b접두사는 tf.string dtype이 유니코드 문자열이 아니라 바이트 문자열임을 나타냄.
UTF-8 기본

tf.cast를 사용하여 문자열 텐서를 숫자로 변환할 수는 없지만, 바이트로 변환한 다음 숫자로 변환 할 수 있음. -> 예제

tf.string dtype은 텐서플로우의 모든 원시 바이트 데이터에 사용


## 희소 텐서

텐서플로우는 tf.sparse.SparseTensor 및 관련 연산을 지원하여 희소 데이터를 효율적으로 저장한다.


### 변수 소개

텐서플로우 변수(tf.Variable)는 프로그램이 조작하는 공유 영구 상태를 표현하는 권장 방법이다.
변수는 tf.Variable클래스를 통해 생성 및 추적된다.
tf.Variable은 ops를 실행하여 값을 변경할 수 있는 텐서.
특정 ops를 사용해 값을 읽고, 수정할 수 있음.
tf.keras와 같은 상위 수준의 라이브러리는 tf.Variable을 사용하여 모델 매개변수를 저장


## 설정

## 변수 만들기

변수를 작성하려면 초기값을 제공해야 한다.
tf.Bariable은 초기화 값과 같은 dtype을 갖는다.

변수는 텐서처럼 보이고 작동하며, tf.Tensor에서 지원하는 데이터 구조
dtype 과 shape를 가지고, numpy로 보낼 수 있다.

변수를 재구성할 수 없지만, 대부분의 텐서 연산은 변수에 대해 작동

tf.Variable.assign을 사용하여 텐서를 재할당할 수 있다.
assign을 호출해도 새로운 텐서를 할당하지 않고, 기존 텐서의 메모리가 재사용된다. -> 메모리 사용의 효율성

텐서와 같은 변수를 사용하는 경우, 지원 텐서에서 작동
새 변수를 만들면 지원 텐서가 복제, 두 변수는 같은 메모리를 공유하지 않음.

## 수명 주기, 이름 지정 및 감시

파이썬 기반인 텐서플로우에서 tf.Variable은 다른 python객체와 같은 수명 주기를 갖는다.
변수에 대한 참조가 없으면 자동으로 할당이 해제 된다.

변수를 추적하고 디버그하는 데 도움이 되는 변수의 이름을 지정할 수 있다.
두 변수에 같은 이름을 지정할 수 있다.

모델을 저장하고 로드할 때 변수 이름이 유지 된다.
기본적으로 모델의 변수는 고유한 변수 이름이 자동으로 지정되므로 원치 않는 한 직접 할당할 필요가 없다.

변수는 구별을 위해 중요하지만, 일부 변수는 구별할 필요가 없다.
생성 시 trainable을 false로 설정하여 변수의 그래디언트를 끌 수 있다.
그래디언트가 필요하지 않은 변수의 예는 훈련 단계 카운터 이다. -> 예제


## 변수 및 텐서 배치하기

더 나은 성능을 위해 tensorflow는 dtype과 호환되는 가장 빠른 기기에 텐서 및 변수를 배치하려고 시도한다.
이는 대부분의 변수가 GPU에 배치됨을 의미한다. 사용 불가한 경우 제외.

GPU가 사용가능한 경우에도 부동 텐서와 변수를 CPU에 배치할 수 있다.
기기 배치 로깅을 켜면 변수가 어디에 배치되었는지 확인할 수 있다.

수동 배치도 가능하지만, 분배 전략을 사용하면 계산을 최적화하는 더 편리하고 확장 가능한 방법이 될 수 있다.

GPU가 있거나 없는 서로 다른 백엔드에서 이 노트북을 실행하면 서로 다른 로깅이 표시. 세션 시작 시 기기 배치 로깅을 켜야 한다. 

한 기기에서 변수 또는 텐서의 위치를 설정하고 다른 기기에서 계산을 수행할 수 있다.
기기 간에 데이터를 복사해야 하므로 지연이 발생.

GPU작업자가 여러 개이지만 변수의 사본이 하나만 필요한 경우에 수행할 수 있다.

tf.config.set_soft_device_placement는 기본적으로 켜져 있기 때문에 GPU가 없는 기기에서 이 코드를 실행하더라도 코드는 계속 실행되고 곱셈 단계는 CPU에서 발생한다.
