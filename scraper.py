
import requests
from bs4 import BeautifulSoup
import os
from time import sleep
import pandas as pd

clmns = ['type', 'loc', 'sqm', 'lvl', 'nbed', 'nbath', 'year', 'price']
house_stats = pd.DataFrame(columns=clmns)

# Supporting Functions
def get_clean_num(string, position):
    return [int(s) for s in string.split() if s.isdigit()][position]

def get_clean_str(string, position):
    return [str(s) for s in string.split() if s.isalpha()][position]


# Main Function
def main():

    # URL = "https://www.xe.gr/property/r/poliseis-katoikion/ChIJRzGst-u7oRQR9_0w_5XaINg_peiraias"
    # r = requests.get(URL)

    # soup = BeautifulSoup(r.content, features="html5lib")

    # with open("output1.html", "w") as file:
    #     file.write(str(soup))


    with open("page{}.html".format(1), "r") as file:
        soup = BeautifulSoup(file, 'html.parser')

    curr_page_properties = soup.find_all("div", {"class":"common-property-ad"})
    for i in range(len(curr_page_properties)):
        instance_title = str(curr_page_properties[i].find("div", {"class":"common-property-ad-title"}).getText())
        instance_type = [str(s) for s in instance_title.split() if s.isalpha()][0]
        square_meters = [int(s) for s in instance_title.split() if s.isdigit()][0]

        price_raw = str(curr_page_properties[i].find("span", {"class":"property-ad-price"}).getText().replace('.',''))
        price_clean = get_clean_num(price_raw, 0)

        try:
            bedrooms_raw = str(curr_page_properties[i].find("div", {"class":"property-ad-bedrooms-container"}).getText()).replace("×","")
            bedrooms_clean = get_clean_num(bedrooms_raw, 0)
        except:
            print('\n\nbedroom problem')
            continue

        try:
            bathrooms_raw = str(curr_page_properties[i].find("div", {"class":"property-ad-bathrooms-container"}).getText()).replace("×","")
            bathrooms_clean = get_clean_num(bathrooms_raw, 0)
        except:
            print('\n\nbathroom problem')
            continue

        try:
            construction_year_raw = str(curr_page_properties[i].find("div", {"class":"property-ad-construction-year-container"}).getText()).replace("×","")
            construction_year_clean = get_clean_num(construction_year_raw, 0)
        except:
            print('\n\nage problem')
            continue
        
        try:
            level_raw = str(curr_page_properties[i].find("span", {"class":"property-ad-level"}).getText()).replace("ος","").replace("+","").replace(",","")
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
            print('\nLevel Problem')
            break

        location_raw = str(curr_page_properties[i].find("span", {"class":"common-property-ad-address"}).getText()).replace("Πειραιάς (","").replace(") | Πώληση κατοικίας","")
        location_clean = ' '.join([str(s) for s in location_raw.split() if s.isalpha()])

        print('\nInstance {}\n'.format(str(i)))
        print('type: ' + str(instance_type))
        print('location: ' + str(location_clean))
        print('square meters: ' + str(square_meters))
        print('level: ' + str(level_clean))
        print('number of bedrooms: ' + str(bedrooms_clean))
        print('number of bathrooms: ' + str(bathrooms_clean))
        print('age: ' + str(construction_year_clean))
        print('price: ' + str(price_clean) + "€")

if __name__ == "__main__":
    main()