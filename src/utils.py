from tqdm import tqdm
import glob
import pandas as pd
import numpy as np



def initialize():
    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', '내부 습도 1 최저', '내부 습도 1 최고',
                    '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

    csv_files = sorted(glob.glob('data/train/train/*/*.csv'))

    temp_csv = pd.read_csv(csv_files[42])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    #featrue 별 min,max 계산
    for csv in tqdm(csv_files[1:]):

        temp_csv = pd.read_csv(csv)[csv_features]
        temp_csv = temp_csv.replace('-', np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr, temp_max], axis=0)
        min_arr = np.min([min_arr, temp_min], axis=0)

    #feature 별 최댓값 최솟값 딕셔너리
    csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}

    return csv_feature_dict


def label_description():
    crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
    disease = {
            '1': {
                'a1': '딸기잿빛곰팡이병 - 확산되지 않도록 꽃과 과실을 제거 후, 벤지미다졸 계열 약제를 살포 후 습도 및 환기에 유의해주세요', 
                'a2': '딸기흰가루병 - 일조량을 높여주고, 유황계 살균제를 살포해주세요',
                'b1': '냉해피해 - 온실 내부 온도를 유지해주세요', 
                'b6': '다량원소결핍 (N) - 질소 함유 비료를 추가해주세요',
                'b7': '다량원소결핍 (P) - 인 함유 비료를 추가해주세요',
                'b8': '다량원소결핍 (K) - 칼륨 함유 비료를 추가해주세요'
            },
            '2': {
                'a5': '토마토흰가루병 - 확산되지 않도록 감염 부위를 제거하고, 유황계 약제를 살포 후 습도를 조절해주세요', 
                'a6': '토마토잿빛곰팡이병 - 감염부위를 제거하고, 일조량을 높인 후 벤지미다졸 계열 살균제를 살포해주세요', 
                'b2': '열과 - 온실 내부 온도를 유지해주세요', 
                'b3': '칼슘결핍 - 칼슘 함유 비료를 추가해주세요',
                'b6': '다량원소결핍 (N) - 질소 함유 비료를 추가해주세요',
                'b7': '다량원소결핍 (P) - 인 함유 비료를 추가해주세요', 
                'b8': '다량원소결핍 (K) - 칼륨 함유 비료를 추가해주세요'
            },
            '3': {
                'a9': '파프리카흰가루병 - 감염부위를 제거하고 유황계 살균제를 사용 후 습도관리를 해주세요', 
                'a10': '파프리카잘록병 - 감염부위를 제거 후 다진계 살균제를 사용 후 물을 관리해주세요', 
                'b3': '칼슘결핍 - 칼슘 함유 비료를 추가해주세요', 
                'b6': '다량원소결핍 (N) - 질소 함유 비료를 추가해주세요', 
                'b7': '다량원소결핍 (P) - 인 함유 비료를 추가해주세요', 
                'b8': '다량원소결핍 (K) - 칼륨 함유 비료를 추가해주세요'
            },
            '4': {
                'a3': '오이노균병 - 에타브론, 메타락실등 살균제를 사용하고 감염부위를 제거해주세요', 
                'a4': '오이흰가루병 - 유황계 살균제를 사용 후 감염부위를 제거 후 환기를 자주 시켜주세요', 
                'b1': '냉해피해 - 온실 내부 온도를 유지해주세요', 
                'b6': '다량원소결핍 (N) - 질소 함유 비료를 추가해주세요', 
                'b7': '다량원소결핍 (P) - 인 함유 비료를 추가해주세요', 
                'b8': '다량원소결핍 (K) - 칼륨 함유 비료를 추가해주세요' 
            },
            '5': {
        'a7': '고추탄저병 - 아졸계 살균제를 이용하여 살균하고 감염부위를 제거 후 환기 개선 및 습도 관리를 해주세요', 
        'a8': '고추흰가루병 - 유황계 살균제를 사용 후 감염부위를 제거 후 적절한 간격으로 물을 주고 습도를 조절해주세요', 
        'b3': '칼슘결핍 - 칼슘 함유 비료를 추가해주세요', 
        'b6': '다량원소결핍 (N) - 질소 함유 비료를 추가해주세요', 
        'b7': '다량원소결핍 (P) - 인 함유 비료를 추가해주세요', 
        'b8': '다량원소결핍 (K) - 칼륨 함유 비료를 추가해주세요'
    },
    '6': {
        'a11': '시설포도탄저병 - 아졸계 살균제를 사용 후 감염 부위를 제거하고 환기 및 습도 조절을 해주세요', 
        'a12': '시설포도노균병 - 포스틸 알루미늄 살균제를 사용하고 감염 부위 제거 후 적절한 간격으로 물과 습도를 조절해주세요', 
        'b4': '일소피해 - 차광막을 설치하고 피해 부위를 제거해주세요', 
        'b5': '축과병 -  적절한 물과 배수 관리를 해주고 피해 부위를 제거해주세요'
    }
    }

    risk = {'1':'초기','2':'중기','3':'말기'}

    label_description_dict = {}

    for key, value in disease.items():
 
        label_description_dict[f'{key}_00_0'] = f'{crop[key]}_정상'
        for disease_code in value:
            for risk_code in risk:
                label = f'{key}_{disease_code}_{risk_code}'
                label_description_dict[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
                
   
    label_encoder = {key:idx for idx, key in enumerate(label_description_dict)}
    label_decoder = {val:key for key, val in label_encoder.items()}


    return label_decoder, label_encoder, label_description_dict

