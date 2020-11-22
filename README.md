# KoGPT2-storyline-generation

Storyline Generation Model based on KoGPT2

한국어 사전학습 언어모델인 KoGPT2를 기반으로 한 스토리라인 생성 모델입니다.

사전학습모델은 SKTAI의 KoGPT2를 사용하였습니다. (https://github.com/SKT-AI/KoGPT2)



## Prerequisites

먼저, Github 저장소를 본인의 디렉토리에 받아주세요.

```
$git clone https://github.com/jucho2725/KOGPT2_storyline_generation.git
```

필수 패키지를 설치하기 위해 다음을 실행해주세요. 

패키지의 버젼이 적힌 것과 다를시 다양한 오류가 발생할 수 있습니다. 

```
$pip install -r requirements.txt
```

## How to train

학습하기 위한 간단한 예제입니다. 

```
$python train.py --data_path your_file
```

자세한 arguments 는 `$python train.py -h` 를 통해 확인해주세요.



모델에서 학습에 사용된 데이터는 네이버 영화 줄거리이며, 크롤링한 데이터 중 장르 정보가 있는 데이터 약 4만7천여 건입니다.

## How to generate

훈련된 모델로 텍스트를 생성하기 위해선 `inference.py` 를 실행해주세요

```
$python inference.py
```

## Try Demo

훈련된 모델은 데모 페이지 링크에서 확인해볼 수 있습니다. (http://115.145.173.133:1994/)

링크에서 'Ver2. KoGPT2-Storyline' 모델을 확인해주세요

(주의: 언제든 데모 페이지 사용이 종료될 수 있습니다.)

![image-20201122185924356](/home/jucho/.config/Typora/typora-user-images/image-20201122185924356.png)

## Things to know

Transformers 에서 데이터 생성 후처리를 위해 generation 관련 util 들을 수정했습니다. `modeling_utils.py` 의 generation 함수에서 확인 가능합니다.

## Contact

관련된 문의는 cju2725@gmail.com 으로 부탁드립니다.

