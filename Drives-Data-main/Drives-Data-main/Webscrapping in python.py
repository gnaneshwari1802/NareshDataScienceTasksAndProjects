# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:25:17 2023

@author: M GNANESHWARI
"""
import requests
from bs4 import BeautifulSoup
url="https://in.indeed.com/jobs?q=data+science&l=&from=searchOnHP&vjk=54a028028b378e84"
response=requests.get(url)
print(response)
print(response.content)
print("*"*108)
page_content=BeautifulSoup(response.content,"html.parser")
print(page_content)
"all the webpage content has got converted into html content"
"from that html content again we need to do scrapping to fetch the specific data"
"right click on the web page and click on the inspect and in it in the elements we can select for any elements and do scrapping"
list1=page_content.select(".jobsearch-ResultList")
print(list1)
jobs=list1[0].find_all("div",class_="job_seen_beacon")
print(jobs)
for j in jobs:
    jobs_title=jobs.find("h2",class_="jobTitle").text
    print("jobs_title"+jobs_title)
    company_name=jobs.find("span",class_="companyName").text
    print("company_name",company_name)
    location=jobs.find("div",class_="companylocation").text
    print("location",location)
print("-"*108)    