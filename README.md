# ML_Stock
Stock Prediction by LSTM(

1. 변수 정의
- 기업명 정의
- 기간 정의

2. 트윗 데이터 생성
- 기간 동안 기업명이 포함된 트윗 크롤링(GetOldTweets3)
- 긍정 단어가 들어간 개수인 Positive 컬럼 추가
- 부정 단어가 들어간 개수인 Negative 컬럼 추가
- (Positive - Negative)인 Adjust 컬럼 추가
- 트윗 데이터 컬럼 : Date, Positive, Negative, Adjust

3. 주가 데이터 생성
- 기간 동안 기업에 대한 주가 데이터 겟(pandas_datareader - Yahoo Finance)
- 전일 대비 최고가 편차에 대한 Gap 컬럼 추가
- 컬럼 : Date, High, Low, Open, Close, Adj_Close, Volumn, Gap

4. 트윗 + 주가 데이터 결합
- 주가 데이터에 존재하는 날짜에 대해서만 트윗 데이터 결합
- LSTM Input 컬럼만 추출
- 컬럼 : Date(Index), High, Low, Open, Close, Volumn, Positive, Negative

5. 데이터 전처리
- MinMaxScaler 적용
- Train/Valid/Test Set 분할
- Batch Data Set(DataLoader) 분할

6. LSTM 모델 생성 및 학습

7. Test Set으로 예측 데이터 생성

8. Test Set과 예측데이터의 일일 등락 일치 여부로 점수 평가

9. 그래프화
