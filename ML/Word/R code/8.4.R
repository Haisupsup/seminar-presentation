#tidyverse 설치
#install.packages("tidyverse")
library(tidyverse)

#kickstarter 데이터 불러오기
kickstarter <- read_csv("C:/Users/user/Desktop/2024-1/R 세미나/kickstarter.csv")
kickstarter

#tidymodels 설치
#install.packages("tidymodels")
library(tidymodels)

#textrecipes 설치
#install.packages("textrecipes")
library(textrecipes)

#rsample 패키지 설치 및 실행
#install.packages("rsample")
library(rsample)
kickstarter_split <- kickstarter %>%
  filter(nchar(blurb) >= 15) %>%
  initial_split()
#이전 강의 데이터 생성하기
kickstarter_train <- training(kickstarter_split)
kickstarter_test <- testing(kickstarter_split)
length(kickstarter_train)
length(kickstarter_test)
#kick_prep 만들기
#prep는 데이터 전처리를 위한 함수
kick_prep <-  prep(kick_rec)
#bake함수를 사용하면 새로운 자료에 recipe 객체에 수행했던 것과 같은 전처리를 수행할 수 있다
kick_train <- bake(kick_prep, new_data = NULL, composition = "matrix")

set.seed(234)
kick_val <- validation_split(kickstarter_train, strata = state)
kick_val

state_analysis <- analysis(kick_val$splits[[1]]) %>% pull(state) #분석을 수행
state_assess <- assessment(kick_val$splits[[1]]) %>% pull(state) #평가를 수행

#전처리된 데이터를 행렬 형태로 반환
kick_analysis <- bake(kick_prep, new_data = analysis(kick_val$splits[[1]]),
                      composition = "matrix")

kick_assess <- bake(kick_prep, new_data = assessment(kick_val$splits[[1]]),
                    composition = "matrix")

#kick_rec 만들기
max_words <- 2e4
max_length <- 30

kick_rec <- recipe(~ blurb, data = kickstarter_train) %>%
  step_tokenize(blurb) %>%
  step_tokenfilter(blurb, max_tokens = max_words) %>%
  step_sequence_onehot(blurb, sequence_length = max_length)

#kick_rec 확인하기
kick_rec

##blurb 열이 단일 예측 변수임을 나타냄, blurb에 대한 tokenization, text filtering, one-hot encoding 진행 

#패키지 설정
library(keras)
library(tensorflow)
# 다운로드한 파일의 경로를 지정합니다.
file_path <- "C://Users//user//Desktop//2024-1//R 세미나//glove.6B.50d.txt"  # 여기서 "path_to_downloaded_file.txt"는 실제 파일의 경로로 대체되어야 합니다.

# 파일을 읽어와서 데이터를 로드합니다.
## header = 파일에 열 이름이 포함되어 있는지 여부를 나타내는 값, 여기서는 False로 파일에 열 이름이 없다고 가정
## sep = 파일의 각 열이 구분되는 구분자를 지정.(빈 공백 문자를 구분자로 사용하고자 하므로 ""로 설정)
glove_data <- read.table(file_path, header = FALSE, sep = " ")

# 로드된 데이터를 확인합니다.
head(glove_data)

library(tibble)

# 데이터프레임을 tibble로 변환하기.
## tibble : 데이터프레임의 확장형이며 데이터프레임보다 사용자 친화적이고 일관된 동작을 제공함.
## as_tibble() : 데이터를 tibble 형식으로 변환하도록 하는 함수.
glove6b_tibble <- as_tibble(glove_data)

# 첫 번째 출력과 동일한 형식으로 데이터를 읽기
## 책에서는 tibble된 glove 데이터를 불러오지만, txt파일은 그렇지 않기에 tibble형식으로 변환

#kick_prep 데이터를 tidy한 형태로 정리
tidy(kick_prep)

#각 변수에 대해 최대 3개의 유일한 값을 유지
tidy(kick_prep, number = 3)

# glove6b_matrix 만들기
glove6b_matrix <- tidy(kick_prep, 3) %>% 
  select(token) %>% # glove6b_matrix에서 token열만 선택
  left_join(glove6b_tibble, by = c("token" = "V1")) %>% # token열을 기준으로 glove6b_tibble와 조인
  mutate(across(-token, replace_na, 0)) %>% # token열을 제외한 모든 열에 대해 결측값을 0으로 대체
  select(-token) %>% # token열을 제외한 모든 열을 선택
  as.matrix() %>% # 데이터를 행렬 형식으로 변환함
  rbind(0, .) # 행렬의 첫 번째 행에 0 벡터를 추가함


#dense_model_pte 만들기
dense_model_pte <- keras_model_sequential() %>% #Sequential모델을 생성
  layer_embedding(input_dim = max_words + 1, #input_dim은 max_words+1한 값
                  output_dim = ncol(glove6b_matrix), #output_dim은 glove6b_matrix의 컬럼 값
                  input_length = max_length) %>% #입력 sequence는 max_length의 길이와 같도록
  layer_flatten() %>% #Flatten레이어를 추가(1차원으로 펼쳐줌)
  layer_dense(units = 32, activation = "relu") %>% #units는 뉴런의 수, activation은 활성화 함수를 의미
  layer_dense(units = 1, activation = "sigmoid") #여기서는 activation으로 sigmoid 사용

#형식 설정
dense_model_pte %>% 
  get_layer(index = 1) %>% #첫 번째 레이어에 설정 
  set_weights(list(glove6b_matrix)) %>% #가중치를 고정
  freeze_weights() 
#optim 설정
dense_model_pte %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) #compile하여 optimizer, loss, metrics를 설정

#train 진행
dense_pte_history <- dense_model_pte %>%
  fit(
    x = kick_analysis, #train data의 입력
    y = state_analysis, #test data의 입력
    batch_size = 512,
    epochs = 20,
    validation_data = list(kick_assess, state_assess), #valid data 설정, 검증 데이터의 입력과 타깃 값을 나타냄
    verbose = FALSE
  )

#evaluation
dense_pte_history

#keras_predict라는 사용자 함수 만들기
keras_predict <- function(model, baked_data, response) {
  predictions <- predict(model, baked_data)[, 1]
  tibble(
    .pred_1 = predictions,
    .pred_class = if_else(.pred_1 < 0.5, 0, 1),
    state = response
  ) %>%
    mutate(across(c(state, .pred_class),            ## create factors
                  ~ factor(.x, levels = c(1, 0))))  ## with matching levels
}
## baked_data는 입력 데이터를 나타냄, model은 예측에 사용될 모델
## predict 함수의 결과는 각 샘플에 대한 예측 확률이므로 저장
## tibble함수를 사용하여 결과를 tibble 형태로 변환(pred_1열에는 예측 확률이 저장, pred_class열에는 예측된 클래스(0 or 1)가 저장)
## across 함수를 사용하여 state열과 pred_class열을 모두 factor로 변환.

#pte_res를 만들어서 factor 0,1 확인
pte_res <- keras_predict(dense_model_pte, kick_assess, state_assess) 
metrics(pte_res, state, .pred_class) #metrics함수로 예측 결과와 실제 클래스 간의 성능 지표 계산

#딥러닝 모델 구현
dense_model_pte2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1,
                  output_dim = ncol(glove6b_matrix),
                  input_length = max_length) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#형식 설정
#freeze_weigths()를 사용하여 가중치를 설정하는 유무의 차이
dense_model_pte2 %>%
  get_layer(index = 1) %>%
  set_weights(list(glove6b_matrix))

#optim 설정
dense_model_pte2 %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#train 진행
dense_pte2_history <- dense_model_pte2 %>% fit(
  x = kick_analysis,
  y = state_analysis,
  batch_size = 512,
  epochs = 20,
  validation_data = list(kick_assess, state_assess),
  verbose = FALSE
)

#evaluation
pte2_res <- keras_predict(dense_model_pte2, kick_assess, state_assess)
metrics(pte2_res, state, .pred_class)
