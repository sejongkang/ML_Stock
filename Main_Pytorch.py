import datetime
import time

import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import Stock_LSTM
import matplotlib.font_manager as fm
from Tweet_Crawling import Tweet_Crawling

def create_stock(name, start, end):

    print('=== Create Stock Data ===')
    # get_download_corp()
    code_df = pd.read_csv('DataSet/corp_list.csv')
    code = code_df.query("name=='{}'".format(name))['code'].to_string(index=False)
    code = code.strip()

    df = pdr.get_data_yahoo(code, start, end)

    stock_data = df.dropna()
    stock_data = stock_data.reset_index().rename(columns={'index': 'Date'})
    stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
    stock_data = stock_data.loc[:, ['Date', 'High', 'Low', 'Open', 'Close']]

    # stock_data = stock_data.drop(stock_data[stock_data['Volume'] == 0].index)

    print('--- Create Daily Stock Gap Data ---')
    #Create Daily Gap
    for i in range(len(stock_data)):
        if not i == 0:
            stock_data.loc[i, 'Gap'] = stock_data.iloc[i, 1] - stock_data.iloc[i - 1, 1]

    return stock_data

def create_crawlings(name, start, end):
    print('=== Create Tweet Data ===')
    df_tweet = pd.read_csv('DataSet/Tweets/{}_twitter_data_{}_to_{}.csv'.format(name, start, end))
    pos_file = open('DataSet/Words/positive_words.txt', 'r', encoding='UTF8')
    neg_file = open('DataSet/Words/negative_words.txt', 'r', encoding='UTF8')
    files = [pos_file, neg_file]
    words_lists = []
    for file in files:
        words_list = []
        for line in file:
            stripped_line = line.strip()
            words_list.append(stripped_line)
        words_lists.append(words_list)
    pos_file.close()
    neg_file.close()
    print('--- Analysis Tweet Data ---')
    for i, twt in df_tweet.iterrows():
        pos_point = 0
        neg_point = 0
        for j in range(len(words_lists)):
            for word in words_lists[j]:
                if word in twt['text']:
                    if j == 0:
                        pos_point = pos_point + 1
                    else:
                        neg_point = neg_point + 1
        df_tweet.loc[i, 'pos'] = pos_point
        df_tweet.loc[i, 'neg'] = neg_point

    df_tweet['Positive'] = df_tweet.groupby(['date'])['pos'].transform('sum')
    df_tweet['Negative'] = df_tweet.groupby(['date'])['neg'].transform('sum')
    df_tweet['Adjust'] = df_tweet['Positive'] - df_tweet['Negative']
    df_tweet = df_tweet.drop_duplicates(subset=['date'])
    df_tweet.rename(columns={'date': 'Date'}, inplace=True)
    # df_tweet = df_tweet.loc[:, ['Date', 'Positive', 'Negative', 'Adjust']]
    df_tweet = df_tweet.loc[:, ['Date', 'Positive', 'Negative']]
    # df_tweet = df_tweet.loc[:, ['Date', 'Adjust']]

    return df_tweet

def scoring(true, pred):
    count = 0
    true = true[:, 0].flatten()
    pred = pred[:, 0].flatten()
    for i in range(1, len(true)):
        true_gap = 1 if true[i] - true[i - 1] > 0 else(-1 if true[i] - true[i - 1] < 0 else 0)
        pred_gap = 1 if pred[i] - pred[i - 1] > 0 else(-1 if pred[i] - pred[i - 1] < 0 else 0)
        if true_gap == pred_gap:
            count = count + 1
    return round(count / len(true) * 100, 1)

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = [data[i + seq_length][0]]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_download_stock(market_type=None):
    if market_type == 'kospi':
        tmp = 'stockMkt'
    else:
        tmp = 'kosdaqMkt'
    download_link = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&marketType=' + tmp
    df = pd.read_html(download_link, header=0)[0]
    return df

def get_download_corp():
    kospi_df = get_download_stock('kospi')
    kospi_df.종목코드 = kospi_df.종목코드.map('{:06d}.KS'.format)
    kosdaq_df = get_download_stock('kosdaq')
    kosdaq_df.종목코드 = kosdaq_df.종목코드.map('{:06d}.KQ'.format)
    code_df = pd.concat([kospi_df, kosdaq_df])
    code_df = code_df[['회사명', '종목코드']]
    code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
    code_df.to_csv('DataSet/corp_list.csv')

if __name__ == "__main__" :

    data_start_time = time.time()

    # Stock-parameters
    stock_name = '삼성전자'
    start = datetime.datetime.strptime("2010-04-01", "%Y-%m-%d")
    end = datetime.datetime.strptime("2020-04-01", "%Y-%m-%d")
    print('=== {} 에서 {} 까지 데이터를 통한 {} 주식 예측 ==='.format(start, end, stock_name))

    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    days_range = []
    for date in date_generated:
        days_range.append(date.strftime("%Y-%m-%d"))

    # Get Tweets
    tweet = Tweet_Crawling(stock_name, days_range)
    tweet.get_tweet()
    tweet.save_tweet()

    # Create Datas
    print('=== Create Data ===')
    twt_data = create_crawlings(stock_name, days_range[0], days_range[-1])
    stock_data = create_stock(stock_name, days_range[0], days_range[-1])

    data = pd.merge(stock_data, twt_data, on='Date', how='left').fillna(0)
    data.index = data['Date']

    # Data Save/Load

    data.to_csv("DataSet/Merged/{}_merged_data_{}_to_{}.csv".format(stock_name, days_range[0], days_range[-1]), index=False)

    data = pd.read_csv("DataSet/Merged/{}_merged_data_{}_to_{}.csv".format(stock_name, days_range[0], days_range[-1]), index_col=0)
    data.index = pd.to_datetime(data.index)

    # Hyper-parameters
    seq_length = 30
    input_size = data.shape[1]
    hidden_size = 256
    num_layers = 2
    num_classes = 1
    batch_size = 20
    num_epochs = 10
    learning_rate = 0.001

    print('=== Split Data ===')

    # Data PreProcessing
    test_data_size = int(round(data.shape[0] * 0.2))
    train_valid_data = data[:-test_data_size]
    test_data = data[-test_data_size:]

    valid_data_size = int(round(train_valid_data.shape[0] * 0.2))
    train_data = train_valid_data[:-valid_data_size]
    valid_data = train_valid_data[-valid_data_size:]

    print('=== Scale Data ===')

    scaler = MinMaxScaler().fit(train_data)

    train_data = scaler.transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_data, seq_length)
    X_valid, y_valid = create_sequences(valid_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    print('=== Create DataSet ===')

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                                  torch.tensor(y_valid, dtype=torch.float32))
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    data_end_time = time.time()
    print(" Create data end.. {0:0.2f} Minutes".format((data_end_time - data_start_time) / 60))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Stock_LSTM.RNN(input_size, hidden_size, num_layers, num_classes, device)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('=== Train Model ===')
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (train_seq, train_price) in enumerate(train_loader):
            train_seq = train_seq.reshape(-1, seq_length, input_size).to(device)
            train_price = train_price.to(device)

            # Forward pass
            outputs = model(train_seq)
            loss = criterion(outputs, train_price)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if valid_loader is not None:
                for j, (valid_seq, valid_price) in enumerate(valid_loader):
                    valid_seq = valid_seq.reshape(-1, seq_length, input_size).to(device)
                    valid_price = valid_price.to(device)

                    with torch.no_grad():
                        pred_price = model(valid_seq)
                        valid_loss = criterion(pred_price.float(), valid_price)

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), valid_loss.item()))
            # if valid_loss.item() < 0.5:
            #     print('Early End')
            #     break

    model_end_time = time.time()
    print(" Train Model end.. {0:0.2f} Minutes".format((model_end_time - data_end_time) / 60))

    torch.save(model, 'Model/stock_model_pytorch.pth')
    print('=== Trained Model Saved ===')

    model = torch.load('Model/stock_model_pytorch.pth')
    print('=== Trained Model Loaded ===')
    print('=== Predict Test Data ===')
    model.eval()
    y_pred = []
    with torch.no_grad():
        for i, (seq, price) in enumerate(test_loader):
            seq = seq.reshape(-1, seq_length, input_size).to(device)
            price = price.to(device)
            y_pred.append([model(seq).item()])
    y_pred = np.array(y_pred)

    tmp = np.zeros((y_train.shape[0], input_size))
    tmp[:, 0:1] = y_train
    y_train = tmp

    tmp = np.zeros((y_valid.shape[0], input_size))
    tmp[:, 0:1] = y_valid
    y_valid = tmp

    tmp = np.zeros((y_test.shape[0], input_size))
    tmp[:, 0:1] = y_test
    y_test = tmp

    tmp = np.zeros((y_pred.shape[0], input_size))
    tmp[:, 0:1] = y_pred
    y_pred = tmp

    print('Train data length : {}'.format(len(X_train)))
    # plt.plot(
    #     data.index[:len(X_train)],
    #     # scaler.inverse_transform(X_train[:, 0, :])[:, 4].flatten(),
    #     X_train[:, 0, :][:, 4].flatten(),
    #     label='Gap'
    # )
    # plt.plot(
    #     data.index[:len(X_train)],
    #     # scaler.inverse_transform(X_train[:, 0, :])[:, 4].flatten(),
    #     X_train[:, 0, :][:, 5].flatten(),
    #     label='Positive'
    # )
    # plt.plot(
    #     data.index[:len(X_train)],
    #     # scaler.inverse_transform(X_train[:, 0, :])[:, 4].flatten(),
    #     X_train[:, 0, :][:, 6].flatten(),
    #     label='Negative'
    # )
    # plt.plot(
    #     data.index[:len(X_train)],
    #     # scaler.inverse_transform(X_train[:, 0, :])[:, 4].flatten(),
    #     X_train[:, 0, :][:, 5].flatten(),
    #     label='Adjust'
    # )
    # plt.plot(
    #     data.index[:len(X_train)],
    #     # scaler.inverse_transform(X_train[:, 0, :])[:, 4].flatten(),
    #     X_train[:, 0, :][:, 0].flatten(),
    #     label='Train'
    # )
    # plt.plot(
    #     data.index[len(train_data) + seq_length:len(train_data) + len(valid_data)],
    #     X_valid[:, 0, :][:, 0].flatten(),
    #     label='Valid'
    # )
    # plt.plot(
    #     data.index[len(train_data) + seq_length:len(train_data) + len(valid_data)],
    #     X_valid[:, 0, :][:, 5].flatten(),
    #     label='Gap'
    # )
    # plt.plot(
    #     data.index[len(train_data) + seq_length:len(train_data) + len(valid_data)],
    #     X_valid[:, 0, :][:, 6].flatten(),
    #     label='Adjust'
    # )
    # plt.plot(
    #     data.index[len(train_data) + seq_length:len(train_data) + len(valid_data)],
    #     y_valid[:, 0].flatten(),
    #     label='Valid'
    # )

    plt.plot(
        data.index[len(train_data) + len(valid_data) + seq_length:len(train_data) + len(valid_data) + len(test_data)],
        # scaler.inverse_transform(y_test)[:, 0].flatten(),
        y_test[:, 0].flatten(),
        label='Real Daily Cases'
    )

    plt.plot(
        data.index[len(train_data) + len(valid_data) + seq_length:len(train_data) + len(valid_data) + len(test_data)],
        # scaler.inverse_transform(y_pred)[:, 0].flatten(),
        y_pred[:, 0].flatten(),
        label='Predicted Daily Cases'
    )

    print('=== Calculate Score ===')
    score = scoring(y_test, y_pred)

    plt.title(stock_name + ' : ' + str(score) + ' %',
              fontproperties=fm.FontProperties(fname='C:\\WINDOWS\\Fonts\\H2GTRM.TTF', size=10))

    plt.savefig('Figure/' + stock_name)
    print(" Total end.. {0:0.2f} Minutes".format((time.time() - data_start_time) / 60))
    plt.legend()
    plt.show()

