## 문제
# 1. 같은 폴더에 있는 new_trigger_algorithm.py를 첨부하고 싶다.
#   new_trigger_algorithm.py를 import 하고 nta 라는 변수로 사용하고 싶다.
## 답

##


# zeroth 데이터 경로
zeroth_path = 'D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean\\test_data_01'

wav_files_list = nta.find_files(filepath=zeroth_path, file_ext=".wav")

# print(wav_files_list)


## 문제
# 2. train_data_set이라는 변수이름으로 파이썬 자료형인 list()를 선언 하시오
## 답

##

for i, one_file in enumerate(wav_files_list):
    one_data = nta.read_wav_file(file_path=one_file)
    one_data = nta.standardize_signal(one_data)
    result = nta.cut_input_signal_for_zeroth(one_data, frame_size=255, shift_size=128)

    result = nta.fit_determined_size(result)
    result = nta.add_noise_data(result)

    train_data_set.append(result)

    ## 문제
    # 3. 동일한 줄에 출력 결과를 갱신하도록 작성하시오.
    ## 답

    ##
    


print(len(train_data_set))
# print(train_data_set)

nta.write_numpy_file(train_data_set, slice_data=200)


















# ## 정답
# 1. import new_trigger_algorithm as nta
# 2. train_data_set = list()
# 3. print("\r{}th file is done...".format(i), end='')

# ## 추가
# 지역 함수를 만들때 사용하는 *args와 **kwargs의 사용법을 익히시오
# 파이썬 자료형인 리스트의 사용법을 익히시오.