import requests
from bs4 import BeautifulSoup
import itertools
from functools import reduce


# # Hitting the top 100 page
link = "http://billboardtop100of.com/1992-2/"
page = requests.get(link)
if (page.status_code == 200):
    print("Hit the page")
else:
    print("Missed it :'(")
soup = BeautifulSoup(page.content, 'html.parser')



# # Extracting top 100

type(soup.find_all('td'))
not_list_top_100 = soup.find_all('td')
print(len(not_list_top_100))
song_list = []
for i in not_list_top_100:
    print (str(i.get_text()))
    song_list.append(i.get_text())
new_song_list = []
for i in range(1,len(song_list),3):
    new_song_list.append(song_list[i] + " - " + song_list[i+1])
print(new_song_list)


# # Writing list to a file
file_song = open('songs_1992.txt', 'w')
for item in new_song_list:
    file_song.write("%s\n" % item)
file_song.close()   


temp_list = ['1','2','3','4']
a = iter(temp_list)
for i in a:
    iter(temp_list)
    print (zip(i,i))
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)
testing = pairwise(temp_list)

for i in testing:
    print (i)
print(iter(temp_list))