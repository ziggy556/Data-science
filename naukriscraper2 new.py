import requests
from bs4 import BeautifulSoup
import urllib
import pandas as pd
from selenium import webdriver
import time

#function for scraping
def extract_job_details_from_result(soup,jobs_title,jobs_comp,sal,loc,exp): 
  for div in soup.find_all(name="div", attrs={"class":"info fleft"}):
      for a in div.find_all(name="a", attrs={"class":"title fw500 ellipsis"}):
          jobs_title.append(a['title']) 
      for ul in div.find_all(name="ul", attrs={"class":"mt-7"}):
          for li in ul.find_all(name="li", attrs={"class":"fleft grey-text br2 placeHolderLi experience"}):
              for span in li.find_all(name="span", attrs={"class":"ellipsis fleft fs12 lh16"}):
                  exp.append(span['title'])
          for li in ul.find_all(name="li", attrs={"class":"fleft grey-text br2 placeHolderLi salary"}):
              for span in li.find_all(name="span", attrs={"class":"ellipsis fleft fs12 lh16"}):
                  sal.append(span['title'])
          for li in ul.find_all(name="li", attrs={"class":"fleft grey-text br2 placeHolderLi location"}):
              for span in li.find_all(name="span", attrs={"class":"ellipsis fleft fs12 lh16"}):
                  loc.append(span['title'])
  for div in soup.find_all(name="div", attrs={"class":"mt-7 companyInfo subheading lh16"}):
      for a in div.find_all(name="a", attrs={"class":"subTitle ellipsis fleft"}):
          jobs_comp.append(a['title'])
  return(jobs_comp,jobs_title,sal,loc,exp)
                  
              
  
#function to get url list
def get_url(base_URL1,url_list,length):
    driver = webdriver.Chrome()
    driver.get(base_URL1)
    time.sleep(5)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for div in soup.find_all(name="div", attrs={"class":"pagination mt-64 mb-60"}):
        for a in div.find_all(name="a", attrs={"class":"fright fs14 btn-secondary br2"}):
            url_list.append('https://www.naukri.com'+a['href'])
    length=length+1
    return (url_list,length)

#To build URL list of all the pages of a website
base_URL = 'https://www.naukri.com/jobs-in-trivandrum?l=trivandrum'
url_list=[]
url_list.append(base_URL)
length=1
while(1):
    (url_list,length)=get_url(url_list[-1],url_list,length)
    if(len(url_list)!=length):
        break
    
#Scraping main function
jobs_title = []
jobs_comp=[]
sal=[]
loc=[]
comp_profile=[]
exp=[]

for i in range(len(url_list)):
    url=url_list[i]
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    (jobs_comp,jobs_title,sal,loc,exp)=extract_job_details_from_result(soup,jobs_title,jobs_comp,sal,loc,exp)
    
#to build dataframe of scraped information
temp=[]
for i in range(3074):
    temp.append((jobs_title[i],jobs_comp[i],sal[i],loc[i],exp[i]))
df=pd.DataFrame(temp,columns=['Job Title','Company Name','Salary','Location','Experience'])
df.to_csv('joblistingsnew.csv')