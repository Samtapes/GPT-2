import datetime as dt
import praw
from psaw import PushshiftAPI
import pandas as pd
# import requests
import ftfy
from newspaper import Article
from env import USERNAME, PASSWORD, CLIENT_ID, CLIENT_SECRET



load_dataframe = False
checkpoint = 2358


def getRedditPosts(limit):
  # Configurating API
  r = praw.Reddit(username=USERNAME,
                  password=PASSWORD,
                  client_id=CLIENT_ID,
                  client_secret=CLIENT_SECRET,
                  user_agent="praw_scrapper_1.0")


  api = PushshiftAPI(r)


  # Searching for posts
  start_epoch=int(dt.datetime(2017, 1, 1).timestamp())

  return list(api.search_submissions(after=start_epoch,
                              subreddit='Brasil',
                              filter=['url','author','title','subreddit', 'score'],
                              limit=limit))

df = None

# Loading DataFrame
if load_dataframe:
  print("=> Load DataFrame")
  df = pd.read_csv('./posts.csv')


# Creating a new one
else:
  print("=> Getting Reddit Posts")
  results = getRedditPosts(limit=1000)

  # DataFrame to store the posts
  df = pd.DataFrame()

  titles = []
  urls = []
  scores = []
  ids = []

  # Getting just posts containing links
  print("=> Getting only posts that contain links")
  banned_sites = ['reddit', 'imgur', 'youtube', 'spotify', 'wikipedia', 'twitter', 'instagram', 'thebloodypitofhorror', 'poa24horas', 'tribunadonorte', 'metropoles', 'ricardoantunes', 'estadao', 'play.google', '12ft.io', 'globoplay', '.pdf', 'netflix.com', 'wattpad.com', 'drive.google', 'bloomberg.com', 'ebay.com', 'google.com/maps', 'msn.com', 'amazom.com', 'google.com/url', 'mongabay.com', '9gag.com', 'jpost.com', 'nytimes.com', 'newsweek.com', 'docs.google']

  for submission in results:
    url_contain_banned_site = any(banned_site in submission.url for banned_site in banned_sites)
    
    if '.com' in submission.url and submission.score >= 3 and not url_contain_banned_site:
      titles.append(submission.title)
      # text.append(submission.selftext)
      urls.append(submission.url)
      scores.append(submission.score)
      ids.append(submission.id)


  # Storing posts
  df['Title'] = titles
  df['Url'] = urls
  df['Id'] = ids
  df['Upvotes'] = scores


  # Saving DataFrame containing good links
  print("=> Saving DataFrame containing good links")
  df.to_csv('./posts.csv', index=False)


# Visualizing DataFrame
print(df.shape)
print(df.head(5))


urls = df['Url']
urls = urls[checkpoint:]


def correcting_n_saving_txt(output):
  corrected_text = ftfy.fix_text(output)
  text_file = open(f"./data/train/content-{idx}.txt", "w", encoding="utf-8")
  n = text_file.write(corrected_text)
  text_file.close()

# Downloading text from the sites urls
print("=> Downloading text from sites")
for idx, url in enumerate(urls, checkpoint):
  output = ''

  try:
    article = Article(url, language='pt', timeout=None)
    article.download()
    article.parse()

    output += f'{article.text}'

    if len(output.split()) >= 513:
      correcting_n_saving_txt(output)
  except Exception as e:
    print(e)
    print(idx)