import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

import math

mpl.rcParams['figure.figsize'] = (8, 6)

print('\n', '그래디언트 기록 제어 예제')
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
  x_sq = x * x
  with t.stop_recording():
    y_sq = y * y
  z = x_sq + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

x = tf.Variable(2.0)
y = tf.Variable(3.0)
reset = True

with tf.GradientTape() as t:
  y_sq = y * y
  if reset:
    # Throw out all the tape recorded so far
    t.reset()
  z = x * x + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])


print('\n', '그래디언트 중지 예제')
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
  y_sq = y**2
  z = x**2 + tf.stop_gradient(y_sq)

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])

print('\n', '사용자 정의 그래디언트 예제')
print('tf.clip_by_norm  예제')
# Establish an identity operation, but clip during the gradient pass
@tf.custom_gradient
# 이 함수 안에 함수가 뭔지 애매하다.
def clip_gradients(y):
  def backward(dy):
    return tf.clip_by_norm(dy, 0.5)
  return y, backward

v = tf.Variable(2.0)
with tf.GradientTape() as t:
  output = clip_gradients(v * v)
print(t.gradient(output, v))  # calls "backward", which clips 4 to 2

print('\n', '여러 테이프 예제')
x0 = tf.constant(0.0)
x1 = tf.constant(0.0)

with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
  tape0.watch(x0)
  tape1.watch(x1)

  y0 = tf.math.sin(x0)
  y1 = tf.nn.sigmoid(x1)

  y = y0 + y1

  ys = tf.reduce_sum(y)

# tf.GradientTape().gradient(target, sources, ...) <- 요모양
tape0.gradient(ys, x0).numpy()   # cos(x) => 1.0

tape1.gradient(ys, x1).numpy()   # sigmoid(x1)*(1-sigmoid(x1)) => 0.25

print('\n', '고계도 그래디언트 예제')
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t2:
  with tf.GradientTape() as t1:
    y = x * x * x

  # Compute the gradient inside the outer `t2` context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t1.gradient(y, x) # -> 왼쪽은 f(x), 오른쪽은 x
d2y_dx2 = t2.gradient(dy_dx, x)

print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0


print('\n', '그래디언트 정규화 예제')
x = tf.random.normal([7, 5])

layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)

with tf.GradientTape() as t2:
  # The inner tape only takes the gradient with respect to the input,
  # not the variables.
  with tf.GradientTape(watch_accessed_variables=False) as t1:
    t1.watch(x)
    y = layer(x)
    out = tf.reduce_sum(layer(x)**2)
  # 1. Calculate the input gradient.
  g1 = t1.gradient(out, x) #-> out을 x로 미분
  # 2. Calculate the magnitude of the input gradient.
  g1_mag = tf.norm(g1)

# 3. Calculate the gradient of the magnitude with respect to the model.
dg1_mag = t2.gradient(g1_mag, layer.trainable_variables)

print([var.shape for var in dg1_mag])

print('\n', '비정형 텐서')


print('\n', '비정형 텐서로 할 수 있는 일 예제')
digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
words = tf.ragged.constant([["So", "long"], ["thanks", "for", "all", "the", "fish"]])
print(tf.add(digits, 3))
print(tf.reduce_mean(digits, axis=1))
print(tf.concat([digits, [[5, 3]]], axis=0))
print(tf.tile(digits, [1, 2]))
print(tf.strings.substr(words, 0, 2))

print(digits[0])       # 첫 번째 행
print(digits[:, :2])   # 각 행의 처음 두 값
print(digits[:, -2:])  # 각 행의 마지막 두 값

# 파이썬 산술 및 비교 연산자를 사용하여 요소별 연산을 수행할 수 있음.
print(digits + 3)
print(digits + tf.ragged.constant([[1, 2, 3, 4], [], [5, 6, 7], [8], []]))
# RaggedTensor의 값으로 요소 별 변환을 수행해야 하는 경우, 함수와 하나 이상의 매개변수를 갖는 tf.ragged.map_flat_values를 사용할 수 있고, RaggedTensor의 값을 변환할 때 적용할 수 있다.
times_two_plus_one = lambda x: x * 2 + 1
print(tf.ragged.map_flat_values(times_two_plus_one, digits))

print('\n', '비정형 텐서 생성 예제')
sentences = tf.ragged.constant([
    ["Let's", "build", "some", "ragged", "tensors", "!"],
    ["We", "can", "use", "tf.ragged.constant", "."]])
print(sentences)

paragraphs = tf.ragged.constant([
    [['I', 'have', 'a', 'cat'], ['His', 'name', 'is', 'Mat']],
    [['Do', 'you', 'want', 'to', 'come', 'visit'], ["I'm", 'free', 'tomorrow']],
])
print(paragraphs)

def print_title(title_sent):
    print('\n', title_sent)
    return

# 각 값이 속하는 행을 알고 있을 때
print_title('tf.RaggedTensor.from_value_rowids 예제')
print(tf.RaggedTensor.from_value_rowids(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    value_rowids=[0, 0, 0, 0, 2, 2, 2, 3]))

# 각 행의 길이를 알고 있을 때
print_title('tf.RaggedTensor.from_row_length 예제')
print(tf.RaggedTensor.from_row_lengths(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    row_lengths=[4, 0, 3, 1]))

# 각 행의 시작과 끝 인덱스를 알고 있을 때
print_title('tf.RaggedTensor.from_row_splits 예제')
print(tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    row_splits=[0, 4, 4, 7, 8]))


print_title('비정형 텐서에 저장할 수 있는 것 예제')
print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]))  # 좋음: 유형=문자열, 랭크=2
print(tf.ragged.constant([[[1, 2], [3]], [[4, 5]]]))        # 좋음: 유형=32비트정수, 랭크=3
try:
  tf.ragged.constant([["one", "two"], [3, 4]])              # 안좋음: 다수의 유형
except ValueError as exception:
  print(exception)
try:
  tf.ragged.constant(["A", ["B", "C"]])                     # 안좋음: 다중첩 깊이
except ValueError as exception:
  print(exception)


print_title('tf.ragged 사용 예시')
queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],
                              ['Pause'],
                              ['Will', 'it', 'rain', 'later', 'today']])

# 임베딩 테이블 만들기
num_buckets = 1024
embedding_size = 4
embedding_table = tf.Variable(
    tf.random.truncated_normal([num_buckets, embedding_size],
                       stddev=1.0 / math.sqrt(embedding_size)))

# 각 단어에 대한 임베딩 찾기
word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
word_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, word_buckets)                  # ①

# 각 문장의 시작과 끝에 마커 추가하기
marker = tf.fill([queries.nrows(), 1], '#')
padded = tf.concat([marker, queries, marker], axis=1)                       # ②

# 바이그램 빌드 & 임베딩 찾기
bigrams = tf.strings.join([padded[:, :-1],
                               padded[:, 1:]],
                              separator='+')                                # ③

bigram_buckets = tf.strings.to_hash_bucket_fast(bigrams, num_buckets)
bigram_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, bigram_buckets)                # ④

# 각 문장의 평균 임베딩 찾기
all_embeddings = tf.concat([word_embeddings, bigram_embeddings], axis=1)    # ⑤
avg_embedding = tf.reduce_mean(all_embeddings, axis=1)                      # ⑥
print(avg_embedding)


print_title('비정형 텐서 형태shape 예제')
print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]).shape)

# tf.RaggedTensor.bounding_shape 메서드를 사용하여 지정된RaggedTensor에 빈틈이 없는 경계 형태를 찾을 수 있음
print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]).bounding_shape())


print_title('비정형 vs 희소 텐서 예제')
ragged_x = tf.ragged.constant([["John"], ["a", "big", "dog"], ["my", "cat"]])
ragged_y = tf.ragged.constant([["fell", "asleep"], ["barked"], ["is", "fuzzy"]])
print(tf.concat([ragged_x, ragged_y], axis=1))

sparse_x = ragged_x.to_sparse()
sparse_y = ragged_y.to_sparse()
sparse_result = tf.sparse.concat(sp_inputs=[sparse_x, sparse_y], axis=1)
print(tf.sparse.to_dense(sparse_result, ''))


print_title('오버로드된 연산자')
x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
y = tf.ragged.constant([[1, 1], [2], [3, 3, 3]])
print(x + y)

x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
print(x + 3)


print_title('비정형 1차원으로 2차원 비정형 텐서 인덱싱 예제')
queries = tf.ragged.constant(
    [['Who', 'is', 'George', 'Washington'],
     ['What', 'is', 'the', 'weather', 'tomorrow'],
     ['Goodnight']])
print(queries[1])

print(queries[1, 2])                # 한 단어
print(queries[1:])                  # 첫 번째 행을 제외한 모든 단어
print(queries[:, :3])               # 각 쿼리의 처음 세 단어
print(queries[:, -2:])              # 각 쿼리의 마지막 두 단어


print_title('비정형 2차원으로 3차원 비정형 텐서 인덱싱 예제')
rt = tf.ragged.constant([[[1, 2, 3], [4]],
                         [[5], [], [6]],
                         [[7]],
                         [[8, 9], [10]]])

print(rt[1])                        # 두 번째 행 (2차원 비정형 텐서)
print(rt[3, 0])                     # 네 번째 행의 첫 번째 요소 (1차원 텐서)
print(rt[:, 1:3])                   # 각 행의 1-3 항목 (3차원 비정형 텐서)
print(rt[:, -1:])                   # 각 행의 마지막 항목 (3차원 비정형 텐서)


print_title('텐서 형 변환 예제')
ragged_sentences = tf.ragged.constant([
    ['Hi'], ['Welcome', 'to', 'the', 'fair'], ['Have', 'fun']])
print(ragged_sentences.to_tensor(default_value=''))

print(ragged_sentences.to_sparse())

x = [[1, 3, -1, -1], [2, -1, -1, -1], [4, 5, 8, 9]]
print(tf.RaggedTensor.from_tensor(x, padding=-1))

st = tf.SparseTensor(indices=[[0, 0], [2, 0], [2, 1]],
                     values=['a', 'b', 'c'],
                     dense_shape=[3, 3])
print(tf.RaggedTensor.from_sparse(st))


print_title('비정형 텐서 평가, 즉시 실행 예제')
rt = tf.ragged.constant([[1, 2], [3, 4, 5], [6], [], [7]])
print(rt.to_list())

# python 인덱싱 사용, 선택한 텐서 조각에 비정형 차원이 없으면 EagerTensor로 변환.
# numpy()메서드를 사용하여 값에 직접 접근할 수 있음.
print(rt[1].numpy())

# tf.RaggedTensor.values 및 tf.RaggedTensor.row_splits 특성 또는 tf.RaggedTensor.row_lengths() 및 tf.RaggedTensor.value_rowids()와 같은 행 분할 메서드를 사용하여 비정형 텐서를 구성 요소로 분해 가능
print(rt.values)
print(rt.row_splits)


print_title('브로드캐스팅 예제')
# x       (2D ragged):  2 x (num_rows)
# y       (scalar)
# 결과     (2D ragged):  2 x (num_rows)
x = tf.ragged.constant([[1, 2], [3]])
y = 3
print(x + y)

# x         (2d ragged):  3 x (num_rows)
# y         (2d tensor):  3 x          1
# 결과       (2d ragged):  3 x (num_rows)
x = tf.ragged.constant(
   [[10, 87, 12],
    [19, 53],
    [12, 32]])
y = [[1000], [2000], [3000]]
print(x + y)

# x      (3d ragged):  2 x (r1) x 2
# y      (2d ragged):         1 x 1
# 결과    (3d ragged):  2 x (r1) x 2
x = tf.ragged.constant(
    [[[1, 2], [3, 4], [5, 6]],
     [[7, 8]]],
    ragged_rank=1)
y = tf.constant([[10]])
print(x + y)

# x      (3d ragged):  2 x (r1) x (r2) x 1
# y      (1d tensor):                    3
# 결과    (3d ragged):  2 x (r1) x (r2) x 3
x = tf.ragged.constant(
    [
        [
            [[1], [2]],
            [],
            [[3]],
            [[4]],
        ],
        [
            [[5], [6]],
            [[7]]
        ]
    ],
    ragged_rank=2)
y = tf.constant([10, 20, 30])
print(x + y)

# 브로드캐스팅 하지 않는 형태의 예
# x      (2d ragged): 3 x (r1)
# y      (2d tensor): 3 x    4  # 뒤의 차원은 일치하지 않습니다.
x = tf.ragged.constant([[1, 2], [3, 4, 5, 6], [7]])
y = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)

# x      (2d ragged): 3 x (r1)
# y      (2d ragged): 3 x (r2)  # 비정형 차원은 일치하지 않습니다.
x = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
y = tf.ragged.constant([[10, 20], [30, 40], [50]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)

# x      (3d ragged): 3 x (r1) x 2
# y      (3d ragged): 3 x (r1) x 3  # 뒤의 차원은 일치하지 않습니다.
x = tf.ragged.constant([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10]]])
y = tf.ragged.constant([[[1, 2, 0], [3, 4, 0], [5, 6, 0]],
                        [[7, 8, 0], [9, 10, 0]]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)


print_title('RaggedTensor 인코딩 예제')
rt = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2],
    row_splits=[0, 4, 4, 6, 7])
print(rt)


print_title('다수의 비정형 차원 예제')
rt = tf.RaggedTensor.from_row_splits(
    values=tf.RaggedTensor.from_row_splits(
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        row_splits=[0, 3, 3, 5, 9, 10]),
    row_splits=[0, 1, 1, 5])
print(rt)
print("형태: {}".format(rt.shape))
print("비정형 텐서의 차원 : {}".format(rt.ragged_rank))

# 팩토리 함수 tf.RaggedTensor.from_nested_row_splits는 row_splits 텐서 목록을 제공하여 다수의 비정형 차원으로 RaggedTensor를 직접 생성하는데 사용.
rt = tf.RaggedTensor.from_nested_row_splits(
    flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10]))
print(rt)


print_title('정형한 내부 차원 예제')
rt = tf.RaggedTensor.from_row_splits(
    values=[[1, 3], [0, 0], [1, 3], [5, 3], [3, 3], [1, 2]],
    row_splits=[0, 3, 4, 6])
print(rt)
print("형태: {}".format(rt.shape))
print("비정형 텐서의 차원 : {}".format(rt.ragged_rank))


print_title('대체 가능한 행 분할 방식 예제')
values = [3, 1, 4, 1, 5, 9, 2, 6]
print(tf.RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8]))
print(tf.RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0]))
print(tf.RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8]))
print(tf.RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8]))
print(tf.RaggedTensor.from_value_rowids(
    values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5))

# RaggedTensor클래스는 이러한 각 행 분할 텐서를 생성
rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
print("      values: {}".format(rt.values))
print("  row_splits: {}".format(rt.row_splits))
print(" row_lengths: {}".format(rt.row_lengths()))
print("  row_starts: {}".format(rt.row_starts()))
print("  row_limits: {}".format(rt.row_limits()))
print("value_rowids: {}".format(rt.value_rowids()))




