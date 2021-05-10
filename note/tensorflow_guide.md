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



### 자동 미분과 그래디언트 테이프

## 그래디언트 테이프

텐서플로우는 자동미분을 위한 tf.GradientTape API를 제공
자동 미분 -> 주어진 입력 변수에 대한 연산의 그래디언트를 계산 하는 것
tf.GradientTape 는 컨텍스트 안에서 실행된 모든 연산을 테이프에 기록 한다.
텐서플로는 후진 방식 자동 미분을 사용해 테이프에 기록된 연산의 그래디언트를 계산한다.

tf.GradientTape 컨텍스트 안에서 계산된 중간값에 대한 그래디언트도 구할 수 있다.

GradientTape.gradient() 메서드가 호출되면 GradientTape에 포함된 리소스가 해제됨.
동일한 연산에 대해 여러 그래디언트를 계산 하려면, 지속성있는 그래디언트 테이프를 생성하면 된다.
이 그래디언트 테이프는 gradient()메서드의 다중 호출을 허용한다.
테이프 객체가 쓰레기 수집될 때 리소스는 해제 된다. 

# 제어 흐름 기록

연산이 실행되는 순서대로 테이프에 기록되기 때문에, 파이썬 제어 흐름(ex. if, while, for)이 자연스럽게 처리된다.

# 고계도 그래디언트

GradientTape 컨텍스트 매니저안에 있는 연산들은 자동미분을 위해 기록.
이 컨텍스트 안에서 그래디언트를 계산한다면 해당 그래디언트 연산 또한 기록되어 진다.
그 결과 똑같은 API가 고계도 그래디언트에서도 잘 작동한다.



### 그래프 및 함수 소개


## 그래프 및 tf.function 소개

tensorflow 코드를 간단하게 변경하여 그래프를 가져오는 방법, 그래프를 저장하고 표현하는 방법, 그리고 그래프를 사용하여 모델의 속도를 높이고 내보내는 방법의 핵심을 알아 본다.

## 그래프란 무엇인가?

지금까지 tensorflow연산이 python에 의해 실행되고 연산별로 결과가 다시 python으로 반환됨을 보여줌.
eager tensorflow는 gpu의 장점을 활용하여 gpu와 tpu에 변수, 텐서 및 연산을 배치할 수 있고, 디버깅도 쉬움.

python에서 텐서 계산을 추출할 수 있다면 그래프로 만들 수 있다.

그래프는 계산 단위를 나타내는 tf.Operation 객체와 연산 간에 흐르는 데이터의 단위를 나타내는 tf.Tensor 객체의 세트를 포함
데이터 구조는 tf.Graph 컨텍스트에서 정의됨. 
그래프는 데이터 구조이므로 원래 python 코드 없이 모두 저장, 실행 및 복원할 수 있음.


## 그래프의 이점

그래프를 사용하면 유연성이 크게 향상됨.
모바일 어플리케이션, 임베디드 기기 및 백엔드 서버와 같은 python 인터프리터가 없는 환경에서 tensorflow 그래프를 사용할 수 있음.
tensorflow는 그래프를 python에서 내보낼 때 저장된 모델의 형식으로 그래프를 사용.

그래프는 쉽게 최적화 되어 컴파일러가 다음과 같은 변환을 수행할 수 있음.
*계산에서 상수 노드를 접어 텐서의 값을 정적으로 추론
*독립적인 계산의 하위 부분을 분리하여 스레드 또는 기기 간에 분할
*공통 하위 표현식을 제거하여 산술 연산을 단순화

변환 및 기타 속도 향상을 수행하기 위한 전체 최적화 시스템으로 Grappler가 있음

그래프는 tensorflow가 빠르게, 병렬로, 그리고 효율적으로 여러 기기에서 실행할 때 아주 유용함.


## 그래프 추적하기

tensorflow에서 그래프를 생성하는 방법은 tf.function을 직접 후출 또는 데코레이터로 사용하는 것

tf.function 화된 함수는 해당 python과 같게 동작하는 python callable이다.
python callable은 특정 클래스(python.eager.def_function.Function)을 갖지만, 비 추적 버전과 같이 동작

tf.function은 호출하는 모든 python 함수를 재귀적으로 추적


## 흐름 제어 및 부작용

흐름 제어 및 루프는 기본적으로 tf.autograph를 통해 tensorflow로 변환.
autograph는 루프 구성의 표준화, 언롤링 및 AST조작을 포함하여 메서드의 조합을 사용

autograph 변환을 직접 호출하여 python이 tensorflow ops로 어떻게 변환 되는지 확인

autograph는 자동으로 if-then, loop, break, return, continue등을 변환


## 속도 향상

tf.function에서 텐서를 사용하는 함수를 래핑하는 것만으로는 코드의 속도를 자동으로 높일 수 없다.

복잡한 계산의 경우, 그래프는 상당한 속도 향상을 제공할 수 있다.
그래프는 python과 기기 간 통신을 줄이고 일부 속도 향상을 수행하기 때문


# 다형 함수

함수를 추적할 때 생성되는 Function 객체는 polymorphic이다.
다형 함수는 하나의 API 뒤에서 몇가지 concrete 함수 그래프를 캡슐화하는 python callable 이다.

이 Function은 모든 종류의 dtypes과 형상에 사용
새로운 인수 서명을 사용하여 호출할 때마다 원래 함수는 새로운 인수로 다시 추적
그런 다음 Function은 cnocrete_function에서의 추적에 해당하는 tf.Graph를 저장
함수가 이미 해당 종류의 인수로 추적되었다면, 미리 추적된 그래프를 얻는다.

*tf.Graph는 계산을 설명하는 원시 휴대용 데이터 구조
*Function은 ConcreteFunctions에 대한 캐싱, 추적, 디스패처 이다.
*ConcreteFunction은 python에서 그래프를 실행 할 수 있는 그래프의 eager 호환 래퍼이다.

# 다형 함수 검사하기

python 함수 my_fuction 에서 tf.function 호출의 결과인 a_function을 검사할 수 있다.
3가지 종류의 인수를 사용하여 A_function을 호출하면 3가지 concrete함수가 생성 -> 예제

추적을 구체적으로 관리하지 않는 한, 일반적으로 여기에 표시된 concrete함수를 직접 후출할 필요는 없다.

# 즉시 실행으로 되돌리기

tf.Graph를 참조하거나 tf.Graph().as_default()를 사용하여 참조하는 경우와 같이 긴 스택 추적을 만나게 될 수도 있다.
tensorflow의 핵심 함수는 Keras의 model.fit()과 같은 그래프 컨텍스트를 사용

그래프에서 디버깅이 까다로워지는 상황에서는 즉시 실행을 사용하여 되돌린 후 디버깅할 수 있다.
***tensorflow framework가 제대로 배포되고 사용되려면 디버깅에 관한 부분도 세심히 개발되었을 것.

즉시 실행을 확인하는 방법
*모델과 레이어를 callable로 직접 호출
*컴파일에서 Keras compile/fit 사용 시, model.compile(run_eagerly=True)사용
*tf.config.run_functions_eagerly(True)를 통해 전역 실행 모드 설정

예제
먼저, eager 없이 모델을 컴파일.
모델은 추적되지 않음.
compile은 손실 함수, 최적화 및 기타 훈련 매개변수만 설정

fit을 호출하고 함수가 추적되고(두 번)eager 효과가 다시 실행되지 않는지 확인

eager에서 단 한번의 epoch만 실행해도 eager 부작용을 두 번 볼 수 있음.

모든 것을 즉시 실행하도록 전역적으로 설정할 수 도 있음.
재추적할 때만 동작
추적된 함수는 추적된 상태로 그래프로 실행


## 추적 및 성능

추적에는 약간의 오버헤드가 발생
작은 함수를 추적하는 것은 빠르지만 큰 모델은 추적하는 데 시간이 오래 걸릴 수 있음.
대형 모델의 훈련에서 처음 몇 번의 epoch가 추적으로 인해 느려질 수 있다

모델의 크기와 관계없이, 빈번한 추적은 피해야 함.
비정상적으로 성능이 저하되고 있음을 발견하면, 실수로 재추적을 하고 있지 않은지 확인

함수가 언제 추적되는지 확인할 수 있도록 eager 부작용을 추가 할 수 있다.(print())
새로운 python 인수가 항상 재추적을 트리거 하기 때문에 추가 재추적이 표시
(원문 : To figure out when your Function is tracing, add a print statement to its code. As a rule of thumb, Function will execute the print statement every time it traces.)



### 모듈, 레이어 및 모델 소개


추상적 모델
*텐서에서 무언가를 계산하는 함수
*훈련에 대한 응답으로 업데이트할 수 있는 일부 변수

## tensorflow 에서 모델 및 레이어 정의하기

레이어는 재사용할 수 있고 훈련 가능한 변수를 가진, 알려진 수학적 구조의 함수
tensorflow 에서 Keras 또는 Sonnet과 같은 레이어 및 모델의 상위 수준 구현 대부분은 같은 기본 클래스인 tf.Module을 기반으로 구축

모듈과 레이어는 "객체"에 대한 딥 러닝 용어 이다.
내부 상태와 해당 상태를 사용하는 메서드가 있다.
python callable 처럼 동작하는 것 외에는 __call__에 특별한 점은 없다.
미세 조정 중 레이어 및 변수 고정을 포함하여 어떤 이유로든 변수의 훈련 가능성을 설정 및 해제할 수 있다.

tf.Module은 tf.keras.layers.Layer 및 tf.keras.Model 의 기본 클래스 이므로 여기에 표시되는 모든 항목도 Keras에 적용.
Keras 레이어는 모듈에서 변수를 수집하지 않으므로 모델은 모듈만 사용하거나 Keras 레이어만 사용해야 함.

tf.Module 을 하위 클래스화함으로써 이 객체의 속성에 할당된 tf.Variable 또는 tf.Module 인스턴스가 자동으로 수집
변수를 저장 및 로드할 수 있으며, tf.Module 모음을 만들 수도 있음.

tf.Module 인스턴스는 tf.Variable 또는 할당된 tf.Module 인스턴스를 재귀적으로 자동 수집
단일 모델 인스턴스로 tf.Module 모음을 관리하고 전체 모델을 저장 및 로드 할 수 있다.

## 변수 생성 연기하기

여기에서 레이어에 대한 입력 및 출력 크기를 모두 정의해야 한다는 것을 알 수 있음.
w변수가 알려진 형상을 가지므로 할당할 수 있다.

특정 입력 형상으로 모듈이 처음 호출될 때까지 변수 생성을 연기하면 입력 크기를 미리 지정할 필요가 없음.

이런 유연성으로 인해 tensorflow 레이어는 종종 입력 및 출력 크기가 아닌 tf.keras.layers.Dense 에서와 같이 출력의 형상만 지정하면 됨


## 가중치 저장하기

tf.Module을 checkpoint 와 SavedModel로 모두 저장할 수 있다.

체크포인트는 데이터 자체와 메타데이터용 인덱스 파일이라는 두 가지 종류의 파일로 구성
인덱스 파일은 실제로 저장된 항목과 체크포인트의 번호를 추적하는 반면 체크포인트 데이터에는 변수 값과 해당 속성 조회 경로가 포함됨.

분산 훈련중에 변수 모음이 샤딩될 수(수평 분할) 있으므로 번호가 매겨짐.


## 함수 저장하기

tensorflow 는 tensorflow serving 및 tensorflow lite 에서와 같이 원래 python 객체 없이 모델을 실행할 수 있으며 tensorflow hub에서 훈련된 모델을 다운로드하는 경우에도 실행할 수 있다.

tensorflow는 python에 설명된 계산을 수행하는 방법을 알아야 하지만 원본 코드는 없다.
그래프를 만들 수 있다.
이 그래프에는 함수를 구현하는 연산 또는 ops가 포함된다.

이 코드가 그래프로 실행되어야 함을 나타내기 위해 @tf.function 데코레이터를 추가하여 위 모델에서 그래프를 정의할 수 있다.

*여기서 그래프란??
tensor의 연산 과정의 컨텍스트를 포함하는 데이터 구조를 말함.


# SavedModel 생성하기

완전히 훈련된 모델을 공유하는 권장 방법은 SavedModel을 사용하는 것
SavedModel에는 함수 모음과 가중치 모음이 모두 포함됨.

예제의 saved_model.pb 파일은 함수형 tf.Graph를 설명하는 프로토콜 버퍼.

모델과 레이어는 실제로 이 표현을 생성한 클래스의 인스턴스를 만들지 않고도 이 표현에서 로드할 수 있다.
대규모 또는 에지 기기에서 제공하는 것과 같이 python 인터프리터가 없거나 또는 원하지 않는 상황 또는 원래 python 코드를 사용할 수 없거나 사용하는 것이 실용적이지 않은 상황에서 사용 -> python이 설치되어 있지 않은 상황에서도 사용할 수 있다.

예제에서 저장된 모델을 로드하여 생성된 new_model은 클래스 지식이 없는 내부 tensorflow 사용자 객체이다. -> SequentialModule_1유형이 아니다.

SavedModel 을 사용하면 tf.Module을 사용하여 tensorflow 가중치와 그래프를 저장한 다음 다시 로드할 수 있다.


## Keras 모델 및 레이어

# Keras 레이어

tf.keras.layers.Layer 는 모든 Keras 레이어의 기본 클래스 이며 tf.Module 에서 상속한다.

부모를 교체한 다음 __call__ 을 call로 변경하여 모듈을 Keras 레이어로 변환할 수 있다.

Keras 레이어에는 다음 섹션에서 설명하는 몇 가지 부기(bookkeeping)를 수행한 다음 call()을 호출하는 고유한 __call__이 있다.


# build 단계

입력 형상(shape)이 확실해질 때까지 변수를 생성하기 위해 기다리는 것이 많은 경우 편리

Keras 레이어에는 레이어를 정의하는 방법에 더 많은 유연성을 제공하는 추가 수명 주기 단계가 있다. -> build()함수에서 정의

build 는 정확히 한 번만 호출되며 입력 형상으로 호출.
변수(가중치)를 만드는 데 사용

예제의 MyDense 레이어를 입력 크기에 맞게 다시 작성할 수 있다.

Keras 레이어에는 다음과 같은 더 많은 추가 기능이 있다.
*선택적 손실
*메트릭 지원
*훈련 및 추론 사용을 구분하기 위한 선택적 training 인수에 대한 기본 지원
*python에서 모델 복제를 허용하도록 구성을 정확하게 저장할 수 있는  get_config 및 from_config 메서드


## Keras 모델

모델을 중첩된 Keras 레이어로 정의

Keras는 tf.keras.Model 이라는 완전한 기능을 갖춘 모델 클래스 제공
tf.keras.layers.Layer 에서 상속되므로 Keras 모델은 Keras 레이어이며 같은 방식으로 사용, 중첩 및 저장 할 수 있음.

거의 동일한 코드로 위에서 SequentialModule을 정의할 수 있고, 다시 __call__ 을 call()로 변환하고 부모를 변경할 수 있음.

추적 변수 및 하위 모듈을 포함하여 같은 기능을 모두 사용할 수 있음.

tf.keras.Model 을 재정의하는 것은 다른 프레임워크에서 모델을 마이그레이션 하는 경우 매우 간단할 수 있다.

기존 레이어와 입력을 간단하게 조합한 모델을 구성하는 경우, 모델 재구성 및 아키텍처와 관련된 추가 기능과 함께 제공되는 함수형 API를 사용하여 시간과 공간을 절약할 수 있다.

큰 차이점은 입력 shape가 함수형 구성 프로세스의 일부로 미리 지정된다는 것.
이 경우 input_shape 인수를 완전히 지정할 필요는 없음.
일부 차원은 None으로 남겨 둘 수 있음.

## Keras 모델 저장하기

Keras 모델에서는 체크포인트를 사용할 수 있고, tf.Module과 같게 보임

Keras 모델은 모듈 tf.saved_models.save()로 저장할 수도 있다.
그러나 Keras 모델에는 편리한 메서드와 기타 기능이 있다.

Keras SavedModels 는 또한 메트릭, 손실 및 옵티마이저 상태를 저장



### 기본 훈련 루프

tensorflow에는 상용구를 줄이기 위해 유용한 추상화를 제공하는 고수준 신경망 API인 tf.Keras API도 포함되어 있다.
그러나 이 가이드는 기본 클래스만 사용한다.

## 기계 학습 문제 해결

*훈련 데이터 얻기
*모델 정의
*손실 함수 정의
*훈련 데이터를 실행하여 이상적인 값에서 손실을 계산
*손실에 대한 기울기를 계산하고 옵티 마이저를 사용하여 데이터에 맞게 변수 조정
*결과 평가

설명을 위해 예제에서는 weight 와 bias, 두 가지 변수가 있는 선형 모델을 개발

x, y가 주어 지면 간단한 선형 회귀를 통해 선의 기울기와 오프셋을 찾으시오.

## 데이터

지도 학습은 입력(x)과 출력(y)을 사용한다.
목표는 입력에서 출력 값을 예측할 수 있도록 쌍을 이룬 입력과 출력에서 학습하는 것.

tensorflow에서 데이터의 각 입력은 거의 항상 텐서로 표현되며 종종 벡터.
지도 학습(supervised learning)에서 출력도 텐서

예제는 선을 따라 가우스 노이즈를 추가하여 합성 된 데이터 이다.


## 모델 정의

tf.Variable 을 사용하여 모델의 모든 가중치를 나타냅니다. 
tf.Variable 은 값을 저장하고 필요에 따라 텐서 형식으로 제공 한다.

tf.Module 을 사용하여 변수와 계산을 캡슐화 한다.
모든 python 객체를 사용할 수 있지만 이렇게 하면 쉽게 저장할 수 있다.

여기서 w 와 b 를 모두 변수로 정의


# 손실 함수 정의

손실 함수는 주어진 입력에 대한 모델의 출력이 목표 출력과 얼마나 잘 일치하는지 측정
훈련 목표는 이러한 차이를 최소화 하는 것
예제는 mean square error 라고 하는 표준 L2 손실을 정의 한다.

# 훈련 루프 정의

*모델을 통해 입력 배치를 전송하여 출력 생성
*출력을 출력(레이블)과 비교하여 손실 계산
*그라디언트 테이프를 사용하여 그라디언트 찾기
*이러한 그라디언트로 변수 최적화

경사 하강 법을 사용하여 모델을 훈련

tf.keras.optimizers 에서 캡처되는 경사 하강 법 체계에는 다양한 변형이 있음.
tf.GradientTape를 사용해서 기본 수학을 사용한 경사하강 전략이 있다.
tf.assign_sub (tf.assign + tf.sub)도 사용한다.


## 동일한 솔루션이지만 Keras를 사용하면

모델을 정의하는 것은 tf.keras.Model 하위 클래스를 tf.keras.Model Keras 모델은 궁극적으로 모듈에서 상속

python 학습 루프를 작성하거나 디버그 하지 않으려는 경우 유용할 수 있음.

model.compile()을 사용하여 매개 변수를 설정하고 model.fit()을 사용하여 학습해야 함.
L2 손실 및 경사 하강 법의 Keras 구현을 바로가기로 사용하는 것은 코드가 적다.

Keras fit 일괄 데이터 또는 전체 데이터 세트를 numpy배열로 예상.



### 고급 자동 미분

tf.GradientTape API의 더 깊고 덜 일반적인 기능에 중점을 둔 가이드



## 그래디언트 기록 제어하기

그래디언트 기록을 중지하려면 GradientTape.stop_recording()을 사용하여 기록을 일시적으로 중단할 수 있다.

모델 중간에서 복잡한 연산을 구별하지 않으려면, 일시 중단이 오버헤드를 줄이는 데 유용할 수 있다.
여기에는 메트릭 또는 중간 결과 계산이 포함될 수 있다.

완전히 다시 시작하려면, reset()을 사용.
그래디언트 테이프 블록을 종료하고 다시 시작하는 것이 일반적으로 읽기 쉽지만, 테이프 블록을 종료하는 것이 어렵거나 불가능한 경우,  reset을 사용.


## 그래디언트 중지

tf.stop_gradient 함수는 테이프 자체에 액세스할 필요 없이 특정 경로를 따라 그래디언트가 흐르는 것을 막는 데 사용할 수 있다.

## 사용자 정의 그래디언트

*작성 중인 새 op에 대해 정의된 그래디언트가 없다.
*기본 계산이 수치적으로 불안정
*정방향 패스에서 값비싼 계산을 캐시하려고 한다.
*그래디언트를 수정하지 않고 값을 수정

새 op를 작성하려면, tf.RegisterGradient를 사용하여 직접 설정할 수 있음.
후자의 세 가지 경우에는 tf.custom_gradient를 사용할 수 있다.

## 여러 테이프

각 테이프는 서로 다른 텐서 세트를 감시

# 고계도 그래디언트

GradientTape 컨텍스트 관리자 내부의 연산은 자동 미분을 위해 기록
해당 컨텍스트에서 그래디언트가 계산되면, 그래디언트 계산도 기록
정확히 같은 API가 고계도 그래디언트에 작동

그래디언트 정규화의 네이티브 구현
*내부 테이프를 사용하여 입력에 대한 출력 그래디언트를 계산
*해당 입력 그래디언트의 크기를 계산
*모델에 대한 해당 크기의 그래디언트를 계산


### 비정형 텐서(Ragged tensor)

## 개요

중첩 가변 길이 목록에 해당하는 tensorflow
균일하지 않은 모양으로 데이터를 쉽게 저장하고 처리
*일련의 영화의 배우들과 같은 가변 길이 기능
*문장이나 비디오 클립과 같은 가변 길이 순차적 입력의 배치
*절, 단락, 문장 및 단어로 세분화된 텍스트 문서와 같은 계층적 입력
*프로토콜 버퍼와 같은 구조화된 입력의 개별 필드

# 비정형 텐서로 할 수 있는 일

비정형 텐서는 수학 연산(예: tf.add 및 tf.reduce_mean), 배열 연산 (예: tf.concat 및 tf.tile), 문자열 조작 작업(예: tf.substr)을 포함하여 수백 가지 이상의 텐서플로 연산에서 지원됨.

python 스타일 인덱싱을 사용하여 비정형 텐서의 특정 부분에 접근할 수 있다.

# 비정형 텐서 생성하기

생성하는 가장 간단한 방법은 tf.ragged.constant를 사용
tf.ragged.constant는 주어진 중첩된python 목록에 해당하는 RaggedTensor 를 빌드

# 비정형 텐서에 저장할 수 있는 것

RaggedTensor의 값은 모두 같은 유형이어야 한다.

# 사용 예시

RaggedTensor를 사용하여 각 문장의 시작과 끝에 특수 마커를 사용하여 가변 길이 쿼리 배치에 대한 유니그램 및 바이그램 임베딩을 생성하고 결합하는 방법을 보여줌


## 비정형 텐서: 정의

# 비정형 및 정형 차원

비정형 텐서는 슬라이스의 길이가 다를 수 있는 하나 이상의 비정형 크기를 갖는 텐서
길이가 모두 같은 차원을 정형 차원이라고 한다.

비정형 텐서의 가장 바깥 쪽 차원은 단일 슬라이스로 구성되므로 슬라이스의 길이가 다를 가능성이 없으므로 항상 균일
비정형 텐서는 균일한 가장 바깥 쪽 차원에 더하여 균일한 내부 차원을 가질 수 도 있음.
[num_sentences, (num_words), embedding_size]에서 num_words는 비정형
[num_documents, (num_paragraphs), (num_sentences), (num_words)] 인 텐서를 사용하여 일련의 구조화된 텍스트 문서를 저장할 수 있다.

# 비정형 텐서 형태 제한

*단일 정형 차원
*하나 이상의 비정형 차원
*0 또는 그 이상의 정형 차원

# 랭크 및 비정형 랭크

비정형 텐서의 총 차원 수를 랭크라 함
비정형 텐서의 비정형 차원 수를 비정형 랭크라 함
그래프 실행 모드 에서 텐서의 비정형 랭크는 생성 시 고정됨.
런타임 값에 의존 할수 없고, 동적으로 변할 수 없음.
잠재적인 비정형 텐서는 tf.Tensor와 tf.RaggedTensor
tf.Tensor 의 비정형 랭크는 0으로 정의 됨.

# 비정형 텐서 형태

비전형 텐서는 괄호로 묶어 표시
단어 임베딩을 저장하는 3차원 비정형텐서의 형태는 [num_sentences, (num_words), embedding_size]로 나타낼 수 있음.


## 비정형 vs 희소 텐서

비정형 텐서는 희소 텐서의 유형이 아니라 불규칙한 형태의 밀집 텐서로 간주되어야 한다.

희소 텐서를 연결하는 것은 다음 예제 처럼 해당 밀집 텐서를 연결하는 것과 같다.


## 오버로드된 연산자

RaggedTensor 클래스는 python 산술 및 비교 연산자를 오버로드하여 기본 요소 별 수학을 쉽게 수행할 수 있음.

오버로드된 연산자는 요소 단위 계산을 수행
이진 연산에 대한 입력은 동일한 형태이거나, 동일한 형태로 브로드캐스팅 할 수 있어야 함.
간단한 확장의 경우, 단일 스칼라가 비정형 텐서의 각 값과 요소 별로 결합됨.

## 인덱싱

비정형 텐서는 다차원 인덱싱 및 슬라이싱을 포함하여 python 스타일 인덱싱을 지원


## 텐서 형 변환

RaggedTensor 클래스는 RaggedTensor와 tf.Tensor 또는 tf.SparseTensors 사이를 변환하는데 사용할 수 있는 메서드를 정의


## 비정형 텐서 평가

# 즉시 실행

비정형 텐서를 python 목록으로 변환하는 tf.RaggedTensor.to_list() 메서드를 사용

# 브로드 캐스팅

브로드 캐스팅은 다른 형태의 텐서가 요소 별 연산에 적합한 형태를 갖도록 만드는 프로세스

호환 가능한 형태를 갖도록 두 개의 입력 x와 y를 브로드캐스팅하는 기본 단계는 다음과 같다.
1. x와 y의 차원수가 동일하지 않은 경우, 외부 차원(크기1)을 차원 수가 동일해질 떄까지 추가한다.
2. x와 y의 크기가 다른 각 차원에 대해:
  * 차원 d에 x 또는 y 의 크기가 1이면, 다른 입력의 크기와 일치하도록 차원 d에서 값을 반복
  * 그렇지 않으면 예외 발생

정형 차원에서 텐서의 크기가 단일 숫자인 경우; 그리고 비정형 차원에서 텐서의 크기가 슬라이스 길이의 목록인 경우


## RaggedTensor 인코딩

비정형텐서는 RaggedTensor 클래스를 사용하여 인코딩
* 가변 길이 행을 병합된 목록으로 연결하는 values 텐서
* 병합된 값을 행으로 나누는 방법을 나타내는 row_splits 벡터, 특히, 행 rt[1]의 값은 슬라이스 rt.values[rt.row_splits[i]:rt.row_splits[i+1]]에 저장


# 다수의 비정형 차원

다수의 비정형 차원을 갖는 비정형 텐서는 values 텐서에 대해 중첩된 RaggedTensor를 사용하여 인코딩
중첩된 각 RaggedTensor는 단일 비정형 차원을 추가

# 정형한 내부 차원

내부 차원이 정형한 비정형 텐서는 values에 다차원 tf.Tensor를 사용하여 인코딩

# 대체 가능한 행 분할 방식

RaggedTensor 클래스는 row_splits를 기본 메커니즘으로 사용하여 값이 행으로 분할되는 방법에 대한 정보를 저장
RaggedTensor는 네 가지 대체 가능한 행 분할 방식을 지원하므로 데이터 형식에 따라 더 편리하게 사용
RaggedTensor는 이러한 추가적인 방식을 사용하여 일부 컨텍스트에서 효율성을 향상

* 행 길이
'row_lenghts'는 '[nrows]'형태의 벡터로, 각 행의 길이를 지정
* 행 시작
'row_starts'는 '[nrows]'형태의 벡터로, 각 행의 시작 오프셋을 지정
'row_splits[:-1]'
* 행 제한
'row_limits'는 '[nrows]'형태의 벡터로 각 행의 정지 오프셋을 지정
'row_splits[1:]'
* 행 인덱스 및 행 수
'value_rowids'는 '[nvals]'모양의 벡터로, 값과 일대일로 대응되며 각 값의 행 인덱스를 지정
'rt[row]'행은 `value_rowids[j]==row`인 `rt.values[j]`값으로 구성
`nrows`는 `RaggedTensor`의 행 수를 지정하는 정수
`nrows`는 뒤의 빈 행을 나타내는데 사용

서로 다른 행 분할 방식의 장점과 단점은 다음과 같습니다:

효율적인 인덱싱: row_splits, row_starts 및 row_limits 방식은 모두 비정형 텐서에 일정한 시간 인덱싱을 가능하게 합니다. value_rowids와 row_lengths 방식은 가능하지 않습니다.

작은 인코딩 크기: 텐서의 크기는 값의 총 수에만 의존하기 때문에 빈 행이 많은 비정형 텐서를 저장할 때 value_rowids 방식이 더 효율적입니다. 반면, 다른 4개의 인코딩은 각 행에 대해 하나의 스칼라 값만 필요하므로 행이 긴 비정형 텐서를 저장할 때 더 효율적입니다.

효율적인 연결: 두 개의 텐서가 함께 연결될 때 행 길이가 변경되지 않으므로 (행 분할 및 행 인덱스는 변경되므로) 비정형 텐서를 연결할 때 row_lengths 방식이 더 효율적입니다.

호환성: value_rowids 방식은 tf.segment_sum과 같은 연산에서 사용되는 분할 형식과 일치합니다. row_limits 방식은 tf.sequence_mask와 같이 작업에서 사용하는 형식과 일치합니다.


### 희소 텐서 작업

많은 0 값을 포함하는 텐서로 작업 할 때 공간 및 시간 효율적인 방식으로 저장하는 것이 중요
희소 텐서는 많은 0값을 포함하는 텐서를 효율적으로 저장하고 처리 할 수 있음.
sparse tensor 는 NLP 어플리케이션에서 데이터 전처리의 일부로 tf-idf와 같은 인코딩 체계에서 광범위하게 사용되며 컴퓨터 비전 애플리케이션에서 어두운 픽셀이 많은 이미지를 전처리하는데 사용

// 추후작업

#### Keras

### 순차모델

## Sequential 모델을 사용하는 경우

Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합

Sequential 모델은 다음의 경우에 적합하지 않다
* 모델에 다중 입력 또는 다중 출력이 있다.
* 레이어에 다중 입력 또는 다중 출력이 있다.
* 레이어 공유를 해야 한다.
* 비선형 토폴로지를 원한다.(예: 잔류 연결, 다중 분기 모델)


## Sequential 모델 생성하기

레이어 목록을 Sequential 생성자에 전달하여 Sequential 모델을 만들 수 있다.

add() 메서드를 통해 Sequential 모델을 점진적으로 작설할 수도 있음.
레이어를 제거하는 pop() 메서드도 있음.
Sequential 모델은 레이어의 리스트와 매우 유사.

Sequential 생성자는 Keras의 모든 레이어 또는 모델과 마찬가지로 name 인수를 허용.
의미론적으로 유의미한 이름으로 TensorBoard 그래프에 주석을 달 때 유용.


## 미리 입력 형상shape 저장하기

일반적으로 Keras의 모든 레이어는 가중치를 만들려면 입력의 형상을 알아야 한다.
처음 레이어를 만들면 가중치가 없다.

가중치weight는 모양이 입력의 형상에 따라 달라지기 때문에 입력에서 처음 호출될 때 가중치를 만듦.

입력 형상이 없는 Sequential 모델을 인스턴스화 할 때는 "빌드"되지 않는다.
가중치가 없음. -> model.weights를 호출하면 오류가 발생
모델에 처음 입력 데이터가 표시되면 가중치가 생성

Input 객체를 모델에 전달하여 모델의 시작 형상을 알 수 있도록 시작
Input 객체는 레이어가 아니므로 model.layers 의 일부로 표시되지 않는다.

간단한 대안은 첫 번째 레이어에 input_shape 인수를 전달하는 것

사전 정의된 입력 모양으로 빌드된 모델은 항상 가중치를 가지며 항상 정의된 출력 형상을 갖는다.
일반적으로 Sequential 모델의 입력 형상을 알고 있는 경우 항상 Sequential 모델의 입력 형상을 지정하는 것이 좋다.


## 일반적인 디버깅 워크플로우 : add() + summary()

새로운 Sequential 아키텍처를 구축할 때는 add() 하여 레이어를 점진적으로 쌓고 모델 요약을 자주 인쇄하는 것이 유용
예를 들어 Conv2D 및 MaxPooling2D 레이어의 스택이 이미지 특성 맵을 다운 샘플링 하는 방법을 모니터링할 수 있다.


## 모델이 완성되면 해야할 일

* 모델을 훈련시키고 평가하며 추론을 실행
* 모델을 디스크에 저장하고 복구
* 다중 GPU를 활용하여 모델의 훈련 속도를 향상

## Sequential 모델을 사용한 특성 추출

Sequential 모델이 빌드되면 Functional API 모델 처럼 동작
이는 모든 레이어가 input 및 output 속성을 갖는다는 것을 의미
Sequential 모델 내의 모든 중간 레이어들의 출력을 추출하는 모델을 빠르게 생성하는 등 깔끔한 작업을 수행


## Sequential 모델을 통한 전이 학습

전이 학습은 모델에서 맨 아래 레이어를 동결하고 맨 위 레이어만 훈련하는 것으로 구성
Sequential 모델이 있고 마지막 모델을 제외한 모든 레이어를 동결하려고 한다고 가정
이 경우 다음과 같이 단순히 model.layers를 반복하고 마지막 레이어를 제외하고 각 레이어에서 layer.trainable = False 를 설정



### 함수형 API

Keras 함수형 API는 tf.keras.Sequential API보다 더 유연한 모델을 생성하는 방법
함수형 API는 비선형 토폴로지, 공유 레이어, 심지어 여러 입력 또는 출력이 있는 모델을 처리할 수 있다.

주요 개념은 딥 러닝 모델은 일반적으로 레이어의 Directed acyclic graph 라는 것
함수형 API는 레이어의 그래프를 빌드하는 방법

"레이어 호출" 동작은 "입력"에서 생성된 레이어로 화살표를 그리는 것과 같다.
입력을 dense 레이어로 "전달"하고 x를 출력으로 가져옴


## 교육, 평가 및 추론

Model 클래스는 내장 훈련 루프 (fit()메서드)와 내장 평가 루프 (evaluate()메서드)를 제공

여기에서 MNIST 이미지 데이터를 로드하고 벡터로 재구성하고 데이터에 모델을 맞추고 테스트 데이터에서 모델을 평가


## 저장 및 직렬화

모델 저장 및 직렬화는 Sequential 모델과 같이 함수형 API를 사용하여 빌드된 모델에 대해 같은 방식으로 작동.
함수형 모델을 저장하는 표준 방법은 model.save() 를 호출하여 전체 모델을 단일 파일로 저장하는 것.

저장된 파일에 포함되는 것
* 모델 아키텍처
* 모델 weight 값
* 모델 훈련 구성
* 옵티마이저 및 상태


## 같은 레이어 그래프를 사용하여 여러 모델 정의하기

함수형 API에서 모델은 레이어 그래프에 입력 및 출력을 지정하여 생성
단일 레이어 그래프를 사용하여 여러 모델을 생성

같은 레이어 스택을 사용하여 두 모델을 인스턴스화
이미지 입력을 16차원 벡터로 변환하는 encoder 모델과 훈련을 위한 End to End audoencoder 모델 => 예제


## 레이어와 마찬가지로 모든 모델은 callable 입니다

Input 또는 또 다른 레이어의 출력에서 모델을 호출함으로써 모델을 마치 레이어와 같이 취급
모델을 호출함으로써 모델의 아키텍처를 재사용할 뿐만 아니라 가중치도 재사용
실례를 위해, 다음은 인코더 모델과 디코더 모델을 만들고 두 번의 호출로 연결하여 자동 인코더 모델을 얻는 자동 인코더 -> 예제

모델은 중첩될 수 있음
모델은 하위 모델을 포함할 수 있음
모델 중첩의 일반적인 사용 사례는 앙상블 기법
모델 세트를 단일 모델로 앙상블하여 예측을 평균화 하는 방법

## 복잡한 그래프 토폴로지 조작

# 여러 입력 및 출력 모델

함수형 API를 사용하면 다중 입력 및 출력을 쉽게 조작할 수 있음.
Sequential API로는 처리할 수 없음.
우선 순위별로 사용자 지정 발급 티켓 순위를 매기고 올바른 부서로 라우팅하는 시스템을 구축하는 경우 모델에는 세가지 입력이 있다.
* 티켓의 제목(텍스트 입력)
* 티켓의 본문(텍스트 입력)
* 사용자가 추가한 모든 태그(범주 입력)

이 모델에는 두 가지 출력이 있음.
* 0과 1사이의 우선 순위 점수(스칼라 시그모이드 출력)
* 티켓을 처리해야하는 부서(부서 세트에 대한 softmax 출력)

예제에 대해, Dataset 객체에 맞춰 호출하면 ([title_data, body_data, tags_data], [priority_targets, dept_targets])와 같은 목록의 튜플 또는 ({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets})와 같은 사전의 튜플이 산출


# 장난감 ResNet 모델

Sequential API가 처리할 수 없는 모델


## 공유 레이어

함수형 API의 또 다른 유용한 용도는 공유 레이어를 사용하는 것
공유 레이어는 같은 모델에서 여러 번 재사용되는 레이어 인스턴스
레이어 그래프에서 여러 경로에 해당하는 특성을 학습

공유 레이어는 종종 비슷한 공간의 입력을 인코딩하는 데 사용
공유 레이어는 서로 다른 입력 간에 정보를 공유할 수 있으며 적은 데이터에서 모델을 훈련
지정된 단어가 입력 중 하나에 표시되면 공유 레이어를 통해 전달하는 모든 입력을 처리하는 데 도움

함수형 API에서 레이어를 공유하려면, 같은 레이어 인스턴스를 여러 번 호출


## 레이어 그래프에서 노드 추출 및 재사용 

조작하는 레이어의 그래프는 정적 데이터 구조이므로 액세스하여 검사할 수 있다.
이를 통해 함수형 모델을 이미지로 플롯할 수 있다.

중간 레이어의 활성화에 액세스하여 다른 곳에 재사용할 수 있다.


## 사용자 정의 레이어를 사용하여 API확장

필요한 것을 찾지 못하면 자신의 레이어를 만들어 API를 쉽게 확장 할 수 있다.
모든 레이어는 Layer 클래스를 서브 클래싱하고 다음을 구현
* 레이어에 의해 수행되는 계산을 지정하는 call 메소드
* build 레이어의 가중치를 생성하는 방법

사용자 정의 레이어에서 직렬화를 지원하려면, 레이어 인스턴스의 constructor 인수를 반환하는 get_config 메서드를 정의

선택적으로, config 사전이 주어진 레이어 인스턴스를 다시 작성할 때 사용되는 클래스 메서드 from_config(cls, config)를 구현

## 함수형 API를 사용하는 경우

Keras 함수형 API를 사용하여 새 모델을 작성하거나 Model 클래스를 직접 하위 클래스화 해야 하는 경우
일반적으로, 함수형 API는 고수준의 쉽고 안전하며, 하위 클래스화 되지 않은 모델에서 지원하지 않는 많은 특성을 가지고 있음


모델 하위 클래스화는 레이어의 방향성 비순환 그래프로 쉽게 표현할 수 없는 모델을 빌드할 때 더 큰 유연성을 제공

# 함수형 API 강점

다음 특성은 Sequential 모델에도 적용되지만, 하위 클래스화된 모델에는 적용 안됨.

* 덜 복잡하다.

* super(MyClass, self).__init__(...), def call(self, ...): 등이 없음.

* 연결 그래프를 정의하면서 모델을 검증
함수형 API에서는 입력 사양이 사전에 작성(Input 사용)
레이어를 호출할 때마다 레이어는 전달된 사양이 해당 가정과 일치하는지 확인하고 그렇지 않은 경우 유용한 오류 메시지를 발생

이를 통해 함수형 API로 빌드할 수 있는 모든 모델이 실행
수렴 관련 디버깅 이외의 모든 디버깅은 실행 중이 아니라 모델 구성 중에 정적으로 발생
컴파일러에서의 유형 검사와 유사

* 함수형 모델은 플롯 및 검사 가능
모델을 그래프로 플롯하고, 이 그래프에서 중간 노드에 쉽게 액세스 할 수 있음.
중간 레이어의 활성화를 추출하여 재사용

* 함수형 모델은 직렬화 또는 복제 가능
함수형 모델은 모드가 아니니 데이터 구조이기 때문에 안전하게 직렬화 할 수 있으며 단일 코드로 저장하여 원본 코드에 액세스하지 않고도 정확히 같은 모델을 다시 만들 수 있다.

하위 클래스화된 모델을 직렬화 하려면 구현자가 모델 레벨에서 get_config() 및 from_config() 메서드를 지정해야 한다.

# 함수형 API 약점

* 동적 아키텍처를 지원하지 않는다.

함수형 API는 모델을 레이어의 DAG로 취급
재귀 네트워크 또는 트리 RNN은 이 가정을 따르지 않으며 함수형 API로 구현할 수 없다.


## 믹스 앤 매치 API 스타일

tf.keras API의 모든 모델은 Sequential 모델, 함수형 모델 또는 처음부터 작성한 하위 클래스화된 모델과 관계없이 서로 상호 작용할 수 있다.

하위 클래스화된 모델 또는 레이어의 일부로 함수형 모델 또는 Sequential 모델을 항상 사용할 수 있다.

다음 패턴 중 하나를 따르는 call 메서드를 구현하는 한 함수형 API에서 하위 클래스화된 레이어 또는 모델을 사용할 수 있다.

사용자 정의 Layer 또는 모델에서 get_config 메서드를 구현하면 작성하는 함수형 모델은 계속 직렬화 가능하고 복제 가능


### 내장 메서드를 사용한 학습 및 평가

## API 개요 : 첫 번째 엔드 투 엔드 예제

데이터를 모델의 내장 훈련 루프로 전달할 때는 numpy 배열 또는 tf.data Dataset 객체를 사용

일반적인 EtoE 플로는 다음과 같이 구성되어 있음.
* 학습
* 원래 교육 데이터에서 생성된 홀드 아웃 세트에 대한 유효성 검사
* 테스트 데이터에 대한 평가


## compile()메소드 :  손실, 메트릭 및 최적화 프로그램 지정

fit()으로 모델을 학습하려면 손실 함수, 최적화 프로그램 및 선택적으로 모니터링 할 일부 메트릭을 지정해야 함.
이것을 compile() 메소드의 인수로 모델에 전달

metrics 인수는 목록이어야 한다.
모델에 여러 개의 출력이 있는 경우 각 출력에 대해 서로 다른 손실 및 메트릭을 지정하고 모델의 총 손실에 대한 각 출력의 기여도를 조정

기본 설정에 만족한다면 대부분의 경우, 최적화, 손실 및 메트릭을 문자열 식별자를 통해 바로 가기로 지정할 수 있음

# 많은 내장 옵티마이저, 손실 및 메트릭을 사용할 수 있음

# 관례 손실

Keras로 커스텀 손실을 제공하는 두 가지 방법
첫번째는 y_true 및 y_pred를 받아 들이는 함수를 만듦

 y_true 및 y_pred 이외의 매개 변수를 사용하는 손실 함수가 필요한 경우 tf.keras.losses.Loss 클래스를 서브 클래스화 하고 다음 두 메소드를 구현할 수 있다.

  * __init__(self) : 손실 함수 호출 중에 전달할 매개 변수를 승인
  * call(self, y_true, y_pred) : 목표(y_true)와 모델 예측(y_pred)을 사용하여 모델의 손실을 계산

  평균 제곱 오차를 사용하려고 하지만 예측 값을 0.5 에서 멀어지게 하는 용어가 추가되었다고 가정해 보면, 이렇게 하면 모델이 너무 자신감이 없는 인센티브가 생겨 과적합을 줄이는 데 도움 -> 예제

# 맞춤 측정 항목

API의 일부가 아닌 메트릭이 필요한 경우 tf.keras.metrics.Metric 클래스를 서브 클래싱하여 사용자 지정 메트릭을 쉽게 만들 수 있다.

* __init__(self) : 여기서 메트릭에 대한 상태 변수를 만듦
* update_state(self, y_true, y_pred, sample_weight=None) 대상 y_true 및 모델 예측 y_pred를 사용하여 상태 변수를 업데이트
* result(self) : 상태 변수를 사용하여 최종 결과를 계산
* reset_states(self): 메트릭의 상태를 다시 초기화

경우에 따라 결과 계산이 많이 들고 주기적으로만 수행되기 때문에 상태 업데이트와 결과 계산은 각각 update_state()와 result() 에서 별도로 유지.


# 표준 서명에 맞지 않는 손실 및 메트릭 처리하기

거의 대부분의 손실과 메트릭은 y_true 및 y_pred 에서 계산할 수 있음.(y_pred가 모델의 출력)
정규화 손실은 레이어의 활성화만 요구

이러한 경우 사용자 정의 레이어의 호출 메서드 내에서 self.add_loss(loss_value)를 호출
추가된 손실은 훈련중 "주요" 손실(compile()로 전달 되는 손실)에 추가

함수형 API에서 model.add_loss(loss_tensor)또는 model.add_metric(metric_tensor, name, aggregation)호출

# 유효성 검사 홀드아웃 세트를 자동으로 분리하기

validation_data 인수를 사용하여 numpy 배열의 튜플(x_val, y_val)을 모델에 전달하여 각 에포크의 끝에서 유효성 검증 손실 및 유효성 검증 메트릭을 평가

인수 validation_split을 사용하여 유효성 검사 목적으로 훈련 데이터의 일부를 자동으로 예약할 수 있음.
인수 값은 유효성 검사를 위해 예약할 데이터 비율을 나타내므로  0보다 크고 1보다 작은 값으로 설정해야 함.
ex) validation_split=0.2 -> 20% / validation_split=0.6 -> 60%
마지막 x%의 샘플을 가져옴.


## tf.data 데이터 세트의 교육 및 평가

데이터가 tf.data.Dataset 객체의 형태로 제공되는 경우

tf.data API는 빠르고 확장 가능한 방식으로 데이터를 로드하고 사전 처리하기 위한 tensorflow 2.0의 유틸리티 세트

Dataset 인스턴스를 메서드 fit(), evaluate() 및 predict()로 직접 전달

데이터 세트는 각 epoch의 끝에서 재설정되므로 다음 epoch에서 재사용할 수 있음

데이터 세트의 특정 배치 수에 대해서만 훈련을 실행하려면 다음 epoch로 이동하기 전에 이 데이터 세트를 사용하여 모델이 실행 해야 하는 훈련 단계의 수를 지정하는 steps_per_epoch 인수를 전달

이렇게 하면 각 epoch가 끝날 때 데이터 세트가 재설정 되지 않고 다음 배치를 계속 가져오게 됨.
무한 반복되는 데이터 세트가 아니라면 결국 데이터 세트의 데이터가 고갈

# 유효성 검사 데이터 집합 사용

fit()에서 Dataset 인스턴스를 validation_data 인수로 전달할 수 있다.

유효성 검사 데이터 세트는 사용 후마다 재설정되므로 항상 에포크에서 에포크까지 동일한 샘플을 평가하게 됨.

인수 validation_split 는 Dataset 객체로 훈련할 때는 지원되지 않는데, 이를 위해서는 데이터세트 샘플을 인덱싱할 수 있어야 하지만 Dataset API에서는 일반적으로 이것이 불가능


## 샘플 가중치 및 클래스 가중치 사용

# 클래스 가중치

이 가중치는 Model.fit()에 대한 class_weight 인수로 사전을 전달하여 설정.
이 사전은(dict()) 클래스 인덱스를 이 클래스에 속한 샘플에 사용해야 하는 가중치

샘플링을 다시 수행하지 않고 클래스의 균형을 맞추거나 특정 클래스에 더 중요한 모델을 훈련시키는데 사용

예를 들어, 데이터에서 클래스 "0"이 클래스 "1"로 표시된 것의 절반인 경우 Model.fit(..., class_weight={0: 1., 1: 0.5})을 사용.

# 샘플 무게

* numpy 데이터에서 학습하는 경우 : sample_weight 인수를 Model.fit()
* tf.data 또는 다른 종류의 반복자에서 훈련 할 때 : Yield(input_batch, label_batch, sample_weight_batch)


## 다중 입력, 다중 출력 모델로 데이터 전달

shape (32, 32, 3) ( (height, width, channels) 입력과 shape (None, 10) 의 시계열 입력 (timesteps, features) -> 예제 
우리의 모델은이 입력들의 조합으로부터 계산 된 두 개의 출력을 가질 것입니다 : "점수"(모양 (1,) )와 5 개의 클래스 (모양 (5,) )에 대한 확률 분포. -> 다중 출력 예제

## 콜백 사용하기

Keras의 콜백은 훈련 중 다른 시점에서 호출되며 다음과 같은 동작을 구현하는 데 사용할 수 있는 객체
* 훈련 중 서로 다른 시점에서 유효성 검사 수행
* 정기적으로 또는 특정 정확도 임계값을 초과할 때 모델 검사점 설정
* 훈련이 정체된 것 처럼 보일 때 모델의 학습 속도 변경
* 훈련이 정체된 것처럼 보일 때 최상위 레이어의 미세 조정
* 교육이 종료되거나 특정 성능 임계 값을 초과한 경우 전자 메일 또는 인스턴트 메시지 알림 보내기

콜백은 fit()에 대한 호출에 리스트로 전달

# 많은 내장 콜백을 사용

* ModelCheckpoint : 주기적으로 모델을 저장
* EarlyStopping : 훈련이 더 이상 유효성 검사 메트릭을 개선하지 못하는 경우 훈련 중단
* TensorBoard : 시각화 할 수 있는 정기적으로 TensorBoard에 쓰기
* CSVLogger : 손실 및 메트릭 데이터를 CSV 파일로 스트리밍

# 자신의 콜백 작성

기본 클래스 keras.callbacks.Callback 을 확장하여 사용자 정의 콜백을 작성할 수 있음.
콜백은 클래스 속성 self.model 통해 연관된 모델에 액세스


## 모델 검사점 설정하기

상대적으로 큰 데이터 세트에 대한 모델을 훈련시킬 떄는 모델의 검사점을 빈번하게 저장하는 것이 중요
이를 수행하는 가장 쉬운 방법은 ModelCheckpoint 콜백을 사용

ModelCheckpoint 콜백을 사용하여 내결함성을 구현
훈련이 무작위로 중단된 경우 모델의 마지막 저장된 상태에서 훈련을 다시 시작


## 학습 속도 일정 사용하기

# 옵티마이저로 일정 전달

옵티 마이저에서 schedule 객체를 learning_rate 인수로 전달하여 정적 학습 속도 감소 스케줄을 쉽게 사용


## 훈련 중 손실 및 메트릭 시각화하기

명령줄에서 TensorBoard 시작
tensorboard --logdir=/full_path_to_your_logs

# TensorBoard 콜백 사용

콜백에서 로그를 작성할 위치만 지정하면 바로 사용


### 하위 클래스화를 통한 새로운 레이어 및 모델 만들기
