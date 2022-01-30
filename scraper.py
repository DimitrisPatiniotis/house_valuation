
import requests
from bs4 import BeautifulSoup
import os
from time import sleep
import pandas as pd
from random import randrange
import datetime
import re

clmns = ['type', 'loc', 'sqm', 'lvl', 'nbed', 'nbath', 'year', 'price']
house_stats = pd.DataFrame(columns=clmns)

# Supporting Functions
def get_clean_num(string, position):
    return [int(s) for s in string.split() if s.isdigit()][position]

def get_clean_str(string, position):
    return [str(s) for s in string.split() if s.isalpha()][position]

def generate_urls(page_nums):
    url_list =[]
    for page in range(2,page_nums):
        url = 'https://www.xe.gr/property/results?page={}&geo_place_id=ChIJRzGst-u7oRQR9_0w_5XaINg&item_type=re_residence&transaction_name=buy'.format(page)
        url_list.append(url)
    print('{} urls generated'.format(page_nums - 1))
    return url_list

def extract_instance(instance):
        instance_title = str(instance.find("div", {"class":"common-property-ad-title"}).getText())
        instance_type = [str(s) for s in instance_title.split() if s.isalpha()][0]
        square_meters = [int(s) for s in instance_title.split() if s.isdigit()][0]

        price_raw = str(instance.find("span", {"class":"property-ad-price"}).getText().replace('.',''))
        price_clean = get_clean_num(price_raw, 0)

        try:
            bedrooms_raw = str(instance.find("div", {"class":"property-ad-bedrooms-container"}).getText()).replace("×","")
            bedrooms_clean = get_clean_num(bedrooms_raw, 0)
        except:
            # print('\n\nbedroom problem')
            return False

        try:
            bathrooms_raw = str(instance.find("div", {"class":"property-ad-bathrooms-container"}).getText()).replace("×","")
            bathrooms_clean = get_clean_num(bathrooms_raw, 0)
        except:
            # print('\n\nbathroom problem')
            return False

        try:
            construction_year_raw = str(instance.find("div", {"class":"property-ad-construction-year-container"}).getText()).replace("×","")
            construction_year_clean = get_clean_num(construction_year_raw, 0)
        except:
            # print('\n\nage problem')
            return False
        
        try:
            level_raw = str(instance.find("span", {"class":"property-ad-level"}).getText()).replace("ος","").replace("+","").replace(",","")
            try:
                level_clean = get_clean_num(level_raw, 0)
            except:
                level_clean = get_clean_str(level_raw, 0)
                if level_clean == 'Ισόγειο':
                    level_clean = 0
                elif level_clean == 'Ημιώροφος':
                    level_clean = 0.5
                elif level_clean == 'Ημιυπόγειο':
                    level_clean = -0.5
        except:
            # print('\nLevel Problem')
            return False

        location_raw = str(instance.find("span", {"class":"common-property-ad-address"}).getText()).replace("Πειραιάς (","").replace(") | Πώληση κατοικίας","")
        location_clean = ' '.join([str(s) for s in location_raw.split() if s.isalpha()])

        house_row = [instance_type, location_clean, square_meters, level_clean, bedrooms_clean, bathrooms_clean, construction_year_clean, price_clean]

        return house_row

def append_page_info(page):
    print(page)
    r = requests.get(page)
    soup = BeautifulSoup(r.text, "html.parser")
    curr_page_properties = soup.find_all("div", {"class":"common-property-ad"})
    for i in range(len(curr_page_properties)):
        extraction_res = extract_instance(curr_page_properties[i])
        if extraction_res:
            house_stats.loc[len(house_stats)] = extraction_res


def find_chars_until_space(str):
    find = re.compile(r"^[^ ]*")
    m = re.search(find, str).group(0)
    return m

# Main Function
def main():

    urls = generate_urls(100)

    for url in urls:
        if len(house_stats) < 1000:
            append_page_info(url)
            print('Page {} done'.format(url))
            print('Have fetched {} total house stats.'.format(len(house_stats)))
            sleep(randrange(20,35))
        else: 
            break
    date = find_chars_until_space(str(datetime.datetime.now()))

    print(house_stats.head(100))
    print(len(house_stats))
    house_stats.to_csv('house_data_{}.csv'.format(date),index=False)

if __name__ == "__main__":
    main()