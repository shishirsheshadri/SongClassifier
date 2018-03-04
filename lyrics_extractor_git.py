
from googleapiclient.discovery import build
import pprint
import requests
from bs4 import BeautifulSoup
import itertools
from functools import reduce


# In[2]:

#Step 1: Set up the api for the custom search engine

my_api_key = "xxxxxxxxxxxxxxxxxx" # please use your own credentials
my_cse_id = "yyyyyyyyyyyyyyyyyyy" # please use your own credentials

#Searcing for the most relavant lyrics

import os
print (os.getcwd())
song_file = open("C:\\Users\\shish_000\\Desktop\\web_scraping\\Training_Set\\songs_1991.txt", "r") #Output of  the lyrics extractor code
song_list = []
for line in song_file:
    song_list.append(line.replace("\n",""))
song_list
len(song_list)

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    try:
        return res['items']
    except:
        print(search_term)
        return (['error'])
def linker(song_name):
    try:
        results = google_search(song_name, my_api_key, my_cse_id, num=10) #Returns top 10 results
    except:
        print(song_name)
        return ('')        
    relavent_link = results[0]
    if(relavent_link == 'error'):
        print(song_name)
        return ('')
    else:
        return str(relavent_link['link'])

link_list = []
for song in song_list:
    link_list.append(linker(song))
len(link_list)
temp_link = link_list
link_list = link_list + temp_link
link_list
link_list.remove('')
#Now since I have the links, I can extract the lyrics one by
def lyrics_returner(link):
    try:
        page = requests.get(link)
    except:
        print(link)
        return (link)
    soup = BeautifulSoup(page.content, 'html.parser')
    verse_length = len(soup.find_all('p', class_='verse'))
    verse_list = soup.find_all('p', class_='verse')
    part_song = " "
    for verse in verse_list:
        part_song = part_song + " " +str(verse.get_text())
    return (part_song.replace("\n"," "))

lyrics_list = []

for link in link_list:
    lyrics_list.append(lyrics_returner(link))

len (lyrics_list)
lyrics_list
file_song = open('C:\\Users\\shish_000\\Desktop\\web_scraping\\Final_training_set\\songs_lyrics_1991_all.txt', 'w')
for item in lyrics_list:
    file_song.write("%s \n" % item.encode('utf-8'))
file_song.close()  
paragraph.text.encode('utf-8')
