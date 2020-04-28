import time
import pandas as pd
import GetOldTweets3 as got
import datetime

class Tweet_Crawling():
    def __init__(self, name, days):
        self.name = name
        self.max = 20
        self.days_range = days
        print("=== 설정된 트윗 수집 기간은 {} 에서 {} 까지 입니다 ===".format(self.days_range[0], self.days_range[-1]))
        print("=== 총 {}일 간의 데이터 수집 ===".format(len(self.days_range)))

    def get_tweet(self):
        print("Collecting data start.. from {} to {}".format(self.days_range[0], self.days_range[-1]))
        start_time = time.time()
        self.twitter_list = []
        count = 0
        while count < len(self.days_range):
            try:
                # 수집 기간 맞추기
                start_date = self.days_range[count]
                end_date = (datetime.datetime.strptime(self.days_range[count], "%Y-%m-%d")
                            + datetime.timedelta(days=1)).strftime("%Y-%m-%d")  # setUntil이 끝을 포함하지 않으므로, day + 1
                # 트윗 수집 기준 정의
                print("=== {} 트윗 수집 시작 ===".format(start_date))

                tweetCriteria = got.manager.TweetCriteria().setQuerySearch(self.name)\
                                                           .setSince(start_date)\
                                                           .setUntil(end_date)\
                                                           .setMaxTweets(self.max)
                tweet = got.manager.TweetManager.getTweets(tweetCriteria)
                for index in tweet:
                    # 메타데이터 목록
                    content = index.text
                    tweet_date = index.date.strftime("%Y-%m-%d")
                    info_list = [tweet_date, content]
                    self.twitter_list.append(info_list)
                print("=== {}(+{})개 수집 완료 ===".format(len(self.twitter_list), len(tweet)))
            except:
                # print("-----------------HTTP Error 429----------------")
                time.sleep(5)
            else:
                count = count + 1
        self.twitter_df = pd.DataFrame(self.twitter_list, columns=["date", "text"])
        print("Collecting data end.. {0:0.2f} Minutes".format((time.time() - start_time)/60))
        print("=== Total num of tweets is {} ===".format(len(self.twitter_df)))

    def save_tweet(self):
        # csv 파일 만들기
        self.twitter_df.to_csv(
            "DataSet/Tweets/{}_twitter_data_{}_to_{}.csv".format(self.name, self.days_range[0], self.days_range[-1]),
            index=False)
        print("=== {} tweets are successfully saved ===".format(len(self.twitter_df)))