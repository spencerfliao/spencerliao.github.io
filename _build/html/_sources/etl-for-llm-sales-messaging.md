
# Scalable Data ETL for AI-Powered Messaging

> *This repository presents a redacted, simulated version of a capstone project developed under NDA with a Canadian AI company. It illustrates generalizable techniques in ETL, information extraction, and preparation for LLM message generation workflows.*



## Overview

This project focuses on building a robust and scalable data pipeline that extracts structured information from public 10-K annual reports. The extracted features are designed to support downstream AI applications, such as personalized sales message generation using LLMs. The pipeline is optimized to handle noisy, unstructured HTML documents, and scale across thousands of companies.

[View Presentation Slides](https://docs.google.com/presentation/d/1E5HGHGIrSbLAhTI5gE3Lx6DWvD-1scxTr3RYDzV9ems/edit)

[View Full Project Report](https://github.com/spencerfliao/etl-for-llm-sales-messaging/blob/1a6e655edab99ad14279b28e673204801ce67f15/report.pdf)

### Project Objectives

- Parse public 10-K filings and extract meaningful company-level attributes
- Design scalable ETL pipelines suitable for integration with a relational database
- Enhance the feature coverage and specificity of company profiles for downstream use
- Prepare data for use in RAG pipelines to power AI-generated outreach messages



### Pipeline Architecture

1. **SEC 10-K Filings (HTML)** – Raw HTML filings as input
2. **HTML Preprocessing & Parsing** – Extracts relevant sections and cleans up formatting
3. **Section Segmentation** – Breaks documents into logical units (e.g. Risk Factors, Business Overview)
4. **Information Extraction Scripts** – Pulls key terms, metrics, and attributes
5. **Feature Dictionary Construction** – Builds structured representations for each company
6. **Structured Output** – Saves as CSV or JSON
7. **(Optional) Database Upload** – Prepared for integration into PostgreSQL


**Team Members:**
- Spencer Liao  
- Haochen He
- Desmond Bai
- Daniel Jimenez

**Supervisors:**
- Miikka Silfverberg | UBC Professor
- Partner Company Team

---


## Pipeline Walkthrough

Imports


```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import process
```

## 1. Search Functionality

Function to quickly get BeautifulSoup from requests


```python
def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"})
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
```

Function that searches for CIK on SEC EDGAR using company name, returns either single or list of results


```python
def search_cik(company_name):
    def fetch_data(key):
        search_url = f'https://www.sec.gov/cgi-bin/browse-edgar?company={key}&match=&filenum=&State=&Country=&SIC=&owner=exclude&Find=Find+Companies&action=getcompany'
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"})
        if response.status_code != 200:
            return "Failed to fetch data"

        soup = BeautifulSoup(response.text, 'html.parser')

        # Check if it's a detailed single company page
        company_info = soup.find('div', class_='companyInfo')
        if company_info:
            cik = company_info.find('a').get_text(strip=True)[:10]
            company_name = company_info.find('span', class_='companyName').text.split(' CIK')[0]
            return cik, company_name

        # Check if it's a list of companies
        table = soup.find('table', class_='tableFile2')
        if table:
            companies = []
            rows = table.find_all('tr')[1:]
            for row in rows:
                cik = row.find_all('td')[0].find('a').get_text(strip=True)
                company_name = row.find_all('td')[1].get_text(strip=True)
                companies.append((cik, company_name))
            return companies

        return None
    
    key = company_name.lower()
    if len(key) > 1 and key[1] == '&':
        key = key.replace('&', '%26')

    result = fetch_data(key)

    if result is None and 'corporation' in key:
        key = key.replace('corporation', 'corp')
        result = fetch_data(key)
    if result is None and 'group' in key:
        key = key.replace('group', '')
        result = fetch_data(key)
    if result is None and ',' in key:
        key = key.replace(',', '')
        result = fetch_data(key)
    if result is None and ' and ' in key:
        key = key.replace(' and ', ' & ')
        result = fetch_data(key)
    if result is None and 'company' in key:
        key = key.replace('company', 'co')
        result = fetch_data(key)
    
    return result

search_cik('Intercom')
```

Fuctions that finds the best match and produces the CIK url


```python
def get_cik_url(company_name):
    search_result = search_cik(company_name)
    if search_result is None:
        return None
    elif isinstance(search_result, tuple):
        cik = search_result[0]
    elif isinstance(search_result, list) and search_result:
        # Use fuzzy matching to find the closest result
        company_names = [result[1] for result in search_result]
        
        # Perform fuzzy matching
        closest_matches = process.extract(company_name, company_names, limit=5)
        
        # Filter and prioritize exact matches or highly similar ones
        best_match = None
        highest_score = 0
        for match in closest_matches:
            name, score = match
            if score > highest_score and ('inc' in name.lower() or 'corp' in name.lower()):
                highest_score = score
                best_match = name
        
        # Initialize cik with a default value
        cik = None
        
        # Find the CIK for the best match
        if best_match:
            for result in search_result:
                if result[1] == best_match:
                    cik = result[0]
                    break

        # If no suitable match is found, fallback to the closest match
        if cik is None:
            closest_match = process.extractOne(company_name, company_names)
            for result in search_result:
                if result[1] == closest_match[0]:
                    cik = result[0]
                    break
    else:
        return None
    
    if cik is None:
        return None
    
    cik_url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=exclude&count=40'
    return cik_url

cik_url = get_cik_url('Blue Buffalo')
cik_url
```

Function to get the url to 10-K filing document in HTML format, with CIK url as key


```python
def get_10k_url(cik_url):
    soup1 = get_soup(cik_url)

    filing_detail_url = None
    html_url = None
    link1 = None
    link2 = None

    # finding the first 10-K link on the CIK page -> filing detail
    table_rows_1 = soup1.find_all('tr')
    for row in table_rows_1:
        if '10-K' in row.text and '10-K/A' not in row.text:
            link1 = row.find('a')['href']
            break
    if link1:
        filing_detail_url = 'https://www.sec.gov' + link1

        soup2 = get_soup(filing_detail_url)
        table_rows_2 = soup2.find_all('tr')

        # finding 10-k doc from filing detail page
        for row in table_rows_2:
            if '10-K' in row.text:
                link2 = row.find('a')['href']
                break
        if link2:
            xbrl_url = 'https://www.sec.gov' + link2
            html_url = xbrl_url.replace('ix?doc=/', '')

    return html_url

html_url = get_10k_url(cik_url)
html_url
```




    'https://www.sec.gov/Archives/edgar/data/1609989/000160998918000019/bluebuffalo12311710-k.htm'




```python
def get_10k_file(html_url):
    
    if html_url is not None:
        response = requests.get(html_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"})
        return response.text
    else:
        return None
```

## 2. Parse Functionality


```python
html_url = get_10k_url(cik_url)
doc_10k = get_10k_file(html_url)
```

Function to find all section title names


```python
from bs4 import BeautifulSoup, Tag

def extract_sections_titles(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = []
    
    # Find all 'table tr' elements
    table_rows = soup.find_all('tr')
    for row in table_rows:
        if 'Business' in row.text:
            parent_node = row.parent
            for sibling in parent_node.children:
                if isinstance(sibling, Tag):  # Check if the sibling is a Tag object
                    a_tags = sibling.find_all('a')
                    combined_text = ""
                    for a_tag in a_tags:
                        text = a_tag.get_text().strip()
                        # Skip page numbers, signatures, and ensure href is assigned only once
                        if text and not text.isdigit() and 'signature' not in text.lower() and 'item' not in text.lower() and 'part' not in text.lower():
                            combined_text += text
                    if combined_text and combined_text not in sections:
                        sections.append(combined_text)
                        
    return sections
```

Function to find all section title names and href links


```python
def extract_section_and_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = {}
    
    # Find all 'table tr' elements
    table_rows = soup.find_all('tr')
    for row in table_rows:
        if 'Business' in row.text:
            parent_node = row.parent
            for sibling in parent_node.children:
                a_tags = sibling.find_all('a')
                combined_text = ""
                href = ""
                for a_tag in a_tags:
                    text = a_tag.get_text().strip()
                    # Skip page numbers, signatures, and ensure href is assigned only once
                    if text and not text.isdigit() and 'signature' not in text.lower() and 'item' not in text.lower() and 'part' not in text.lower():
                        combined_text += text
                        if not href:
                            href = a_tag['href'].strip('#')
                if combined_text:
                    sections[combined_text] = href
                        
    return sections
```


```python
section_links = extract_section_and_links(doc_10k)
sections_df = pd.DataFrame(list(section_links.items()), columns=['Section Title', 'Link'])
sections_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Section Title</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business</td>
      <td>sCFD89D1EFF2E5293B86B54AFC1B3E429</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Risk Factors</td>
      <td>sE0B3F0F9C141501D99BC795B94F63D64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Unresolved Staff Comments</td>
      <td>s9302553BB168589C93C7FC833AF6EA4F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Properties</td>
      <td>sECEB3774BEDD5B739B854E5E8489C0D4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Legal Proceedings</td>
      <td>sA147E4A7EE1C5F59A27CF898FA0C3F54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mine Safety Disclosures</td>
      <td>s4D666224DFE7509C91EDA75AC0A2B315</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Market for the Registrant’s Common Equity, Rel...</td>
      <td>s810C1C726DBF523C81382B1D16F1148D</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Selected Financial Data</td>
      <td>sE916212FBD485BA9857CCDA87D452815</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Management’s Discussion and Analysis of Financ...</td>
      <td>sDD8D7E29282A5C5B828C887406C7ED20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Quantitative and Qualitative Disclosures about...</td>
      <td>s1A0401606D2857BFACCDB8FDD3658E17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Financial Statements and Supplementary Data</td>
      <td>s6DF8710BCDEE51A6AB502170A885F9FF</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Changes in and Disagreements with Accountants ...</td>
      <td>s02ED851F21BA50D486C1D6ECD1F0FA61</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Controls and Procedures</td>
      <td>sCFFD547FE36258C683D583AC369EE63B</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Other Information</td>
      <td>s339305EF066354CBBB088654E4F9BD0D</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Directors, Executive Officers and Corporate Go...</td>
      <td>s02399A9B0F2D5371BF807EABE25D5936</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Executive Compensation</td>
      <td>s6D7683D9DF635408924F618A48A5ED33</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Security Ownership of Certain Beneficial Owner...</td>
      <td>sDBE6D9382657532A8ABC1CAD62D0DAF1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Certain Relationships and Related Transactions...</td>
      <td>s69AAF3B2AC9354DBBF2A80C810F435B6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Principal Accounting Fees and Services</td>
      <td>s781F6D53FBE55CEB97ECE64895F4350C</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Exhibits and Financial Statement Schedules</td>
      <td>sC1D9600246F352848540A930C26B671B</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Form 10-K Summary</td>
      <td>s7AA497D298E85E7B941EC150E95ED4A6</td>
    </tr>
  </tbody>
</table>
</div>



Function to find all section title and content


```python
def extract_sections(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = {}

    table_rows = soup.find_all('tr')
    links_to_process = []

    for row in table_rows:
        if 'Business' in row.text or 'Risk Factors' in row.text:  # Example section titles
            parent_node = row.parent
            for sibling in parent_node.children:
                if isinstance(sibling, Tag):
                    a_tags = sibling.find_all('a')
                    combined_text = ""
                    href = ""
                    for a_tag in a_tags:
                        text = a_tag.get_text().strip()
                        if text and not text.isdigit() and 'signature' not in text.lower() and 'item' not in text.lower() and 'part' not in text.lower():
                            combined_text += text
                            if not href:
                                href = a_tag['href'].strip('#')

                    if combined_text and href:
                        links_to_process.append((combined_text, href))

    for i, (combined_text, href) in enumerate(links_to_process):
        section_div = soup.find('div', id=href)
        if not section_div:
            section_div = soup.find('div', attrs={'name': href})
        if not section_div:
            section_div = soup.find('a', attrs={'name': href})
            if section_div:
                section_div = section_div.parent

        section_content = ""
        if section_div:
            next_node = section_div.find_next_sibling()
            next_href = links_to_process[i + 1][1] if i + 1 < len(links_to_process) else None
            while next_node and ('id' not in next_node.attrs or ('id' in next_node.attrs and next_node['id'] != next_href)):
                if next_node.name and next_node.name != 'hr':
                    section_content += next_node.get_text(separator=' ', strip=True) + "\n"
                next_node = next_node.find_next_sibling()
        
        sections[combined_text] = section_content.strip()

    return sections
```


```python
section_dict = extract_sections(doc_10k)
sections_df = pd.DataFrame(list(section_dict.items()), columns=['Section Title', 'Content'])
sections_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Section Title</th>
      <th>Content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business</td>
      <td>ITEM 1. BUSINESS\nProposed Merger with General...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Risk Factors</td>
      <td>ITEM 1A. RISK FACTORS\nYou should carefully co...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Unresolved Staff Comments</td>
      <td>ITEM 1B. UNRESOLVED STAFF COMMENTS\nNone.\n\nI...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Properties</td>
      <td>ITEM 2. PROPERTIES\nThe following table sets f...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Legal Proceedings</td>
      <td>ITEM 3. LEGAL PROCEEDINGS\nWe are a party to a...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mine Safety Disclosures</td>
      <td>ITEM 4. MINE SAFETY DISCLOSURES\nNot applicabl...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Market for the Registrant’s Common Equity, Rel...</td>
      <td>ITEM 5. MARKET FOR THE REGISTRANT'S COMMON EQU...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Selected Financial Data</td>
      <td>ITEM 6. SELECTED FINANCIAL DATA\n\nThe followi...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Management’s Discussion and Analysis of Financ...</td>
      <td>ITEM 7. BLUE BUFFALO PET PRODUCTS, INC. MANAGE...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Quantitative and Qualitative Disclosures about...</td>
      <td>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSU...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Financial Statements and Supplementary Data</td>
      <td>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Changes in and Disagreements with Accountants ...</td>
      <td>ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCO...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Controls and Procedures</td>
      <td>ITEM 9A. CONTROLS AND PROCEDURES\n\nDisclosure...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Other Information</td>
      <td>ITEM 9B. OTHER INFORMATION\n\nNone.\n\n\n92\n\...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Directors, Executive Officers and Corporate Go...</td>
      <td>ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND COR...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Executive Compensation</td>
      <td>ITEM 11. EXECUTIVE COMPENSATION\nThe informati...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Security Ownership of Certain Beneficial Owner...</td>
      <td>ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFIC...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Certain Relationships and Related Transactions...</td>
      <td>ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRA...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Principal Accounting Fees and Services</td>
      <td>ITEM 14. PRINCIPAL ACCOUNTING FEES AND SERVICE...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Exhibits and Financial Statement Schedules</td>
      <td>ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHE...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Form 10-K Summary</td>
      <td>ITEM 16. FORM 10-K SUMMARY\n\nNone.\n\n97\n\n\...</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Evaluation

Importing company names from database


```python
import json
import urllib.parse
import pandas as pd
from ast import literal_eval
from sqlalchemy import create_engine

with open('/Users/SFL/Documents/1 Learn/UBC/Capstone/credentials.json') as f:
    login = json.load(f)
    
username = login['user']
password = urllib.parse.quote(login['password'])
host = login['host']
port = login['port']

conn = create_engine(f'postgresql://{username}:{password}@{host}:{port}/postgres')

orgs = pd.read_sql_query(
    """
    SELECT DISTINCT organization
    FROM rocketbrew_0424.address_book_supplement
    WHERE organization_name <> 'NONE'
    """,
    conn
    )["organization"]

orgs = [literal_eval(org) for org in orgs]

company_urls = [(org["name"],org["website_url"]) for org in orgs]

data = pd.DataFrame(company_urls,columns=["names","urls"])
```


```python
url_df = pd.read_csv('/Users/SFL/Documents/1 Learn/UBC/Capstone/data/training_urls.csv')
url_df.columns = ['urls']
merged_df = pd.merge(url_df, data, left_on='urls', right_on='urls', how='left')
train_names = merged_df['names'].tolist()
train_names = list(set(train_names))
```


```python
all_cik_results = []
for item in tqdm(train_names):
    result = get_cik_url(item)
    all_cik_results.append(result)
```

    100%|██████████| 3294/3294 [12:52<00:00,  4.27it/s]  



```python
valid_cik_results = [result for result in all_cik_results if result is not None]
```


```python
all_10k_results = []
for item in tqdm(valid_cik_results):
    result = get_10k_url(item)
    all_10k_results.append(result)
```

    100%|██████████| 956/956 [02:53<00:00,  5.51it/s]



```python
valid_10k_results = [result for result in all_10k_results if result is not None]
```


```python

```

Evaluating 10-K parsing algorithm


```python
parsed_10k = []
for item in tqdm(valid_10k_results):
    html_content = get_10k_file(item)
    result = extract_sections(html_content)
    parsed_10k.append(result)
```

    100%|██████████| 187/187 [14:25<00:00,  4.63s/it]



```python
backup = parsed_10k
```

Visualizing Section Title Stats


```python
import matplotlib.pyplot as plt

section_counts = [len(document) for document in parsed_10k]

plt.figure(figsize=(16, 8))
plt.hist(section_counts, bins=range(min(section_counts), max(section_counts) + 2), edgecolor='black')
plt.title('Histogram of Number of Section Titles per 10-K Document')
plt.xlabel('Number of Sections')
plt.ylabel('Frequency')
plt.xticks(range(min(section_counts), max(section_counts) + 1))
plt.show()
```


    
![png](10K%20Filing_files/10K%20Filing_39_0.png)
    



```python
from collections import Counter

all_titles = [title for document in parsed_10k for title in document.keys()]
title_counts = Counter(all_titles)
top_titles = title_counts.most_common(50)
top_titles_list = [title for title, count in top_titles]
total_documents = len([d for d in parsed_10k if d])
top_titles_percentages = [(title, count / total_documents * 100) for title, count in top_titles]

titles, percentages = zip(*top_titles_percentages)
plt.figure(figsize=(10, 8))
plt.barh(titles, percentages)
plt.xlabel('Percentage of Documents (%)')
plt.ylabel('Section Titles')
plt.title('Top Section Titles in 10-K Documents')
plt.gca().invert_yaxis()
plt.show()

top_titles_list
```


    
![png](10K%20Filing_files/10K%20Filing_40_0.png)
    





    ['Legal Proceedings',
     'Controls and Procedures',
     'Risk Factors',
     'Other Information',
     'Business',
     'Unresolved Staff Comments',
     'Properties',
     'Executive Compensation',
     'Financial Statements and Supplementary Data',
     'Mine Safety Disclosures',
     'Directors, Executive Officers and Corporate Governance',
     'Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters',
     'Certain Relationships and Related Transactions, and Director Independence',
     'Form 10-K Summary',
     'Management’s Discussion and Analysis of Financial Condition and Results of Operations',
     'Quantitative and Qualitative Disclosures About Market Risk',
     'Changes in and Disagreements with Accountants on Accounting and Financial Disclosure',
     'Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities',
     'Disclosure Regarding Foreign Jurisdictions that Prevent Inspections',
     'Cybersecurity',
     'Principal Accountant Fees and Services',
     'Exhibits and Financial Statement Schedules',
     '[Reserved]',
     'Principal Accounting Fees and Services',
     'Exhibits, Financial Statement Schedules',
     'Quantitative and Qualitative Disclosures about Market Risk',
     'Reserved',
     'Changes in and Disagreements With Accountants on Accounting and Financial Disclosure',
     'Selected Financial Data',
     "Management's Discussion and Analysis of Financial Condition and Results of Operations",
     "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
     'Certain Relationships and Related Transactions and Director Independence',
     'Exhibit and Financial Statement Schedules',
     'Notes to Consolidated Financial Statements',
     'Competition',
     'Exhibit Index',
     'C.Cybersecurity',
     'Consolidated Statements of Cash Flows',
     'Consolidated Balance Sheets',
     'Report of Independent Registered Public Accounting Firm',
     'General',
     'Overview',
     'EXHIBIT INDEX',
     'Disclosure Regarding Foreign Jurisdictions That Prevent Inspections',
     'Information about our Executive Officers',
     'Forward-Looking Statements',
     'Intellectual Property',
     'Available Information',
     'Business.',
     'Unresolved Staff Comments.']



Compiling all data into dataframe


```python
df_cik = pd.DataFrame({
    'Company Name': train_names,
    'CIK URL': all_cik_results
})
df_10k = pd.DataFrame({
    'CIK URL': valid_cik_results,
    '10K URL': all_10k_results
})
df_inital = pd.merge(df_cik, df_10k, on='CIK URL', how='left').drop_duplicates()
df_sections = pd.DataFrame({
    '10K URL': valid_10k_results,
    'Content': parsed_10k
})
df = pd.merge(df_inital, df_sections, on='10K URL', how='left').drop_duplicates(subset=['Company Name', 'CIK URL', '10K URL'])
df = df.where(pd.notna(df), None)
```

Statistics of Evaluation


```python
count_all = df.shape[0]
count_cik = df['CIK URL'].notna().sum()
count_10k = df['10K URL'].notna().sum()
count_sections = df['Content'].apply(lambda x: len(x.items()) > 0 if x is not None else False).sum()
count_all, count_cik, count_10k, count_sections
print(f"Test Size: {count_all}")
print(f"CIK Count: {count_cik}, Among All: {count_cik/count_all:.3f}")
print(f"10K Count: {count_10k},  Among All: {count_10k/count_all:.3f}")
print(f"Section>0: {count_sections},  Among 10K: {count_sections/count_10k:.3f}")
```

    Test Size: 3294
    CIK Count: 956, Among All: 0.290
    10K Count: 187,  Among All: 0.057
    Section>0: 144,  Among 10K: 0.770


Exporting sample result data


```python
df[df['Content'].apply(lambda x: 20 < len(x.items()) and len(x.items()) < 35 if x is not None else False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }



```python
df[['Company Name', 'Content']].to_csv('/Users/SFL/Documents/1 Learn/UBC/Capstone/sample_10k_data.csv', index=False)
```

Error analysis


```python
df[df['Content'].apply(lambda x: len(x.items()) == 0 if x is not None else False)]['10K URL'].tolist()
```


```python
df[df['Content'].apply(lambda x: len(x.items()) == 0 if x is not None else False)].iloc[0]['10K URL']
```





```python

```

### Post-Processing of sections


```python
# List of essential sections
essential_sections = [
    'Business', 'Risk Factors', 'Quantitative and Qualitative Disclosures About Market Risk', 'Properties',
    'Legal Proceedings', 'Controls and Procedures', 'Financial Statements and Supplementary Data', 
    'Executive Compensation', 'Directors, Executive Officers and Corporate Governance', 
    'Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters', 
    'Management’s Discussion and Analysis of Financial Condition and Results of Operations', 
    'Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities', 
    'Cybersecurity', 'Selected Financial Data', 'Competition', 
    'Certain Relationships and Related Transactions, and Director Independence', 
    'Changes in and Disagreements with Accountants on Accounting and Financial Disclosure', 
    'Disclosure Regarding Foreign Jurisdictions that Prevent Inspections', 
    'Principal Accountant Fees and Services', 'Exhibits and Financial Statement Schedules'
]
```

Fuzzy similarity matching


```python
from fuzzywuzzy import process, fuzz

def find_best_match(key, choices):
    best_match = process.extractOne(key, choices, scorer=fuzz.ratio)
    if best_match[1] >= 80:  # Only consider it a match if the score is 80 or higher
        return best_match[0]
    return None

# Function to restructure the content with fuzzy matching
def restructure_content(content):
    if content is None:
        return None
    structured_content = {}
    for key, value in content.items():
        match = find_best_match(key, essential_sections)
        if match:
            structured_content[match] = value
    return structured_content

df['Structured'] = df['Content'].apply(restructure_content)
df['Structured'] = df['Structured'].apply(lambda x: None if x == {} else x)
```

    WARNING:root:Applied processor reduces input query to empty string, all comparisons will have score 0. [Query: '.']



```python
df[df['Structured'].apply(lambda x: len(x.items()) > 0 if x is not None else False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }



```python
df[df['Structured'].apply(lambda x: len(x) > 0 and all(v for v in x.values()) if x is not None else False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company Name</th>
      <th>CIK URL</th>
      <th>10K URL</th>
      <th>Content</th>
      <th>Structured</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>Signify Health</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/182818...</td>
      <td>{'BASIS OF PRESENTATION': 'Basis of Presentati...</td>
      <td>{'Business': 'Item 1. Business.

Overview

Sig...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>View, Inc.</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/181185...</td>
      <td>{'Business': 'Item 1. Business
Corporate Histo...</td>
      <td>{'Business': 'Item 1. Business
Corporate Histo...</td>
    </tr>
    <tr>
      <th>118</th>
      <td>R1 RCM</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/147259...</td>
      <td>{'[Reserved]': 'Item 6. [Reserved]

40


Item ...</td>
      <td>{'Quantitative and Qualitative Disclosures Abo...</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Keurig Dr Pepper Inc.</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/141813...</td>
      <td>{'Business': 'ITEM 1.   BUSINESS
OUR COMPANY
K...</td>
      <td>{'Business': 'ITEM 1.   BUSINESS
OUR COMPANY
K...</td>
    </tr>
    <tr>
      <th>210</th>
      <td>Oscar Health</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/156865...</td>
      <td>{'Forward-Looking Statements': 'Table of Conte...</td>
      <td>{'Business': 'Item 1. Business

OUR BUSINESS

...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3185</th>
      <td>Nurix Therapeutics</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/154959...</td>
      <td>{'Business': 'Item 1. Business
When used in th...</td>
      <td>{'Business': 'Item 1. Business
When used in th...</td>
    </tr>
    <tr>
      <th>3206</th>
      <td>RPX Corporation</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/150943...</td>
      <td>{'Business': 'Item 1. Business.

Overview
RPX ...</td>
      <td>{'Business': 'Item 1. Business.

Overview
RPX ...</td>
    </tr>
    <tr>
      <th>3247</th>
      <td>Synopsys Inc</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/883241...</td>
      <td>{'Business': 'Item 1. Business
Company and Seg...</td>
      <td>{'Business': 'Item 1. Business
Company and Seg...</td>
    </tr>
    <tr>
      <th>3286</th>
      <td>Prosper Marketplace</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/141626...</td>
      <td>{'Business': 'ITEM 1.  BUSINESS
Overview
Our v...</td>
      <td>{'Business': 'ITEM 1.  BUSINESS
Overview
Our v...</td>
    </tr>
    <tr>
      <th>3291</th>
      <td>Tenet Healthcare</td>
      <td>https://www.sec.gov/cgi-bin/browse-edgar?actio...</td>
      <td>https://www.sec.gov/Archives/edgar/data/70318/...</td>
      <td>{'Business': 'ITEM 1. BUSINESS
OVERVIEW
Tenet ...</td>
      <td>{'Business': 'ITEM 1. BUSINESS
OVERVIEW
Tenet ...</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 5 columns</p>
</div>




```python
df[df['Structured'].apply(lambda x: len(x.items()) > 0 if x is not None else False)].iloc[2, 4]
```




    {'Business': '',
     'Risk Factors': '',
     'Cybersecurity': '',
     'Properties': '',
     'Legal Proceedings': '',
     'Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities': '',
     'Management’s Discussion and Analysis of Financial Condition and Results of Operations': '',
     'Quantitative and Qualitative Disclosures About Market Risk': '',
     'Financial Statements and Supplementary Data': '',
     'Changes in and Disagreements with Accountants on Accounting and Financial Disclosure': '',
     'Controls and Procedures': '',
     'Disclosure Regarding Foreign Jurisdictions that Prevent Inspections': '',
     'Directors, Executive Officers and Corporate Governance': '',
     'Executive Compensation': '',
     'Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters': '',
     'Certain Relationships and Related Transactions, and Director Independence': '',
     'Principal Accountant Fees and Services': '',
     'Exhibits and Financial Statement Schedules': ''}



Visualizing post-processing stats


```python
all_titles = [title for document in df['Structured'].dropna() for title in document.keys()]
title_counts = Counter(all_titles)
top_titles = title_counts.most_common(50)
total_documents = len(df['Structured'].dropna())
top_titles_percentages = [(title, count / total_documents * 100) for title, count in top_titles]
titles, percentages = zip(*top_titles_percentages)

plt.figure(figsize=(10, 8))
plt.barh(titles, percentages)
plt.xlabel('Percentage of Documents (%)')
plt.ylabel('Section Titles')
plt.title('Top Section Titles in 10-K Documents')
plt.gca().invert_yaxis()
plt.show()
```


    
![png](10K%20Filing_files/10K%20Filing_60_0.png)
    



```python

```

## Experimenting with LLM


```python
from meta_ai_api import MetaAI
ai = MetaAI()
```


```python
link = 'https://www.sec.gov/Archives/edgar/data/778972/000143774914003821/fbmi20131231_10k.htm'

ai.prompt(message=f"{link} Search the text of the 10-K filing for the 'Risk Factors' item and provide its content; if not found, return 'None'.")['message']
```




    'Risk Factors\nThe risks and uncertainties described below are not the only ones we face. Additional risks and uncertainties not presently known to us or that we currently deem immaterial may also impair our business operations. If any of the following risks occur, our business, financial condition, and results of operations could be materially harmed.\nRisks Related to Our Business\nWe may not be able to sustain our revenue growth rate.\nWe may not be able to maintain our profitability.\nWe may not be able to compete effectively in the highly competitive markets in which we operate.\nWe may not be able to continue to develop and launch new and innovative products and services.\nWe may not be able to maintain our brand and reputation.\nWe may not be able to attract and retain skilled employees.\nWe may not be able to manage our growth effectively.\nWe may not be able to maintain our global operations effectively.\nWe may not be able to comply with increasingly complex and changing laws and regulations.\nWe may be subject to legal proceedings and regulatory actions.\nWe may not be able to maintain our intellectual property rights.\nWe may be subject to security breaches and cyber attacks.\nWe may not be able to manage our relationships with third-party vendors and suppliers.\nWe may not be able to maintain our information technology systems and infrastructure.\nWe may be subject to disruptions in the global economy and financial markets.\nWe may not be able to maintain our credit rating.\nWe may be subject to interest rate risks.\nWe may be subject to foreign exchange risks.\nWe may be subject to other risks related to our international operations.\nRisks Related to Our Financial Condition\nWe have a significant amount of debt and may not be able to meet our debt obligations.\nWe may not be able to generate sufficient cash to fund our operations and capital expenditures.\nWe may not be able to access the capital markets on favorable terms.\nWe may be subject to covenants and restrictions in our debt agreements.\nRisks Related to Our Industry\nWe operate in a highly competitive market and may not be able to compete effectively.\nWe may not be able to adapt to changes in the market for online and mobile advertising.\nWe may not be able to adapt to changes in the market for online and mobile commerce.\nWe may not be able to adapt to changes in the market for digital payments.\nWe may not be able to adapt to changes in the market for artificial intelligence and machine learning.\nPlease note that this is a summary of the Risk Factors section of the 10-K filing and is not intended to be a comprehensive or exhaustive list of all the risks and uncertainties facing Facebook, Inc. (now known as Meta Platforms, Inc.).\n'




```python
test_result = {}
for section in tqdm(essential_sections):
    content = ai.prompt(message=f"{link} Search the text of the 10-K filing for the '{section}' item and provide its content; if not found, return 'None'.")['message']
    test_result[section] = content
test_result
```

    100%|██████████| 20/20 [05:14<00:00, 15.73s/it]





    {'Business': "The 'Business' item from the 10-K filing is:\nItem 1. Business.\nMeta Platforms, Inc., (“Facebook” or the “Company”) is a technology company that is headquartered in Menlo Park, California. We are a global community of people who use Facebook, Instagram, WhatsApp, or each of their respective family of products. Our products empower people to connect, share, discover, and communicate with each other on mobile devices and personal computers. We have the following reportable segments:\nAdvertising (Facebook and Instagram)\nPayment Processing Fees (Facebook and WhatsApp)\nConsumer Hardware (Oculus)\nOther (Oculus and other products)\nWe had 2.50 billion monthly active users (MAUs) as of December 31, 2013.\nNone of the information contained in this filing constitutes an offer to sell or a solicitation of an offer to buy any securities of the Company.\nLet me know if you need any further assistance!\n",
     'Risk Factors': "Risk Factors\nThe following are some of the key factors that could affect our business, financial condition, and results of operations:\nGeneral Economic and Industry Conditions: Economic downturns, industry downturns, or changes in market conditions could hurt our business.\nCompetition: We face intense competition in the market, which could hurt our business.\nRegulatory Risks: Changes in laws, regulations, or policies could hurt our business.\nUser Engagement: If our users lose interest in our platform, our business could suffer.\nData Privacy and Security: If we fail to protect our users' data, our business could suffer.\nDependence on Key Personnel: Our business is dependent on our key personnel, and the loss of any of them could hurt our business.\nIntellectual Property Risks: If we are unable to protect our intellectual property, our business could suffer.\nLitigation Risks: We are subject to various legal proceedings, which could hurt our business.\nGlobal Expansion Risks: Our international expansion may not be successful.\nCurrency Fluctuations: Changes in currency exchange rates could hurt our business.\nDependence on Third-Party Services: We rely on third-party services, and if they are not available, our business could suffer.\nPlease note that these are just some of the risk factors, and the actual filing may contain additional or different risks.\n",
     'Quantitative and Qualitative Disclosures About Market Risk': 'Quantitative and Qualitative Disclosures About Market Risk\nThe following information is from Item 7A of the 10-K filing:\nMarket Risk\nWe are exposed to market risk related to changes in interest rates, foreign currency exchange rates, and equity and commodity prices.\nInterest Rate Risk\nOur exposure to interest rate risk relates primarily to our investment portfolio. We manage our exposure to interest rate risk through our investment strategy and by maintaining a diversified investment portfolio.\nForeign Currency Exchange Risk\nWe have international operations and are exposed to foreign currency exchange risk related to changes in exchange rates between the U.S. dollar and other currencies. We manage our exposure to foreign currency exchange risk through our hedging strategy.\nEquity Price Risk\nOur exposure to equity price risk relates primarily to our investments in privately held companies. We manage our exposure to equity price risk through our investment strategy and by maintaining a diversified investment portfolio.\nCommodity Price Risk\nWe are not directly exposed to commodity price risk, as we do not engage in commodity-based transactions.\nPlease note that this is a summary of the relevant section and not the full content of the 10-K filing.\n',
     'Properties': 'Properties\nWe lease our principal properties, including our headquarters, in Menlo Park, California. We also lease properties in various other locations in the United States and abroad for our engineering teams, data centers, and offices for our sales and operations teams.\nAs of December 31, 2013, we had:\n1.08 million square feet of leased space in Menlo Park, California, which primarily supports our headquarters and engineering teams;\n431,000 square feet of leased space in New York City, New York, which primarily supports our sales and operations teams;\n342,000 square feet of leased space in Seattle, Washington, which primarily supports our engineering teams;\n275,000 square feet of leased space in San Francisco, California, which primarily supports our engineering teams;\n236,000 square feet of leased space in Los Angeles, California, which primarily supports our sales and operations teams;\n174,000 square feet of leased space in London, United Kingdom, which primarily supports our sales and operations teams;\n163,000 square feet of leased space in Dublin, Ireland, which primarily supports our sales and operations teams;\n135,000 square feet of leased space in Palo Alto, California, which primarily supports our engineering teams;\n103,000 square feet of leased space in Austin, Texas, which primarily supports our sales and operations teams; and\n85,000 square feet of leased space in Tel Aviv, Israel, which primarily supports our engineering teams.\nWe believe our facilities are sufficient for our current needs and do not anticipate any difficulties in obtaining additional space as needed.\n',
     'Legal Proceedings': "Legal Proceedings\nFrom time to time, we are involved in legal proceedings and claims that arise in the ordinary course of our business. These proceedings and claims may relate to a wide range of matters, including intellectual property, advertising, consumer protection, data privacy, employment, and commercial disputes.\nWe are currently involved in several legal proceedings, including the following:\nIn re Facebook, Inc. Securities Litigation: In July 2012, a putative class action lawsuit was filed against us in the United States District Court for the Southern District of New York. The lawsuit alleges that our IPO prospectus and Registration Statement contained false and misleading statements regarding our financial condition and prospects. We believe these claims are without merit and intend to vigorously defend ourselves.\nYahoo! Inc. v. Facebook, Inc.: In November 2012, Yahoo! Inc. filed a lawsuit against us in the United States District Court for the Northern District of California. The lawsuit alleges that we infringe on certain Yahoo! patents related to online advertising and website customization. We believe these claims are without merit and intend to vigorously defend ourselves.\nRemijas v. Facebook, Inc.: In August 2013, a putative class action lawsuit was filed against us in the United States District Court for the Northern District of California. The lawsuit alleges that our practice of scanning the content of private messages for advertising purposes violates the Electronic Communications Privacy Act and California's Invasion of Privacy Act. We believe these claims are without merit and intend to vigorously defend ourselves.\nWe are also subject to various other legal proceedings and claims that arise in the ordinary course of our business. We may incur significant legal expenses in connection with these proceedings, and we may be required to pay significant damages or settlement amounts if we are unsuccessful in our defense.\nPlease note that the text is a snapshot of the company's legal proceedings as of December 31, 2013, and may not reflect the current status of these cases.\n",
     'Controls and Procedures': 'Controls and Procedures\nAs of December 31, 2013, our management, with the participation of our Chief Executive Officer and Chief Financial Officer, evaluated the effectiveness of our disclosure controls and procedures (as defined in Rules 13a-15(e) and 15d-15(e) under the Securities Exchange Act of 1934). Based on this evaluation, our Chief Executive Officer and Chief Financial Officer concluded that as of December 31, 2013, our disclosure controls and procedures were effective.\nOur management is responsible for establishing and maintaining adequate internal control over financial reporting (as defined in Rules 13a-15(f) and 15d-15(f) under the Securities Exchange Act of 1934). Our management, with the participation of our Chief Executive Officer and Chief Financial Officer, evaluated the effectiveness of our internal control over financial reporting as of December 31, 2013. In making this assessment, our management used the criteria set forth by the Committee of Sponsoring Organizations of the Treadway Commission (COSO) in Internal Control—Integrated Framework (1992 framework).\nBased on this evaluation, our management, including our Chief Executive Officer and Chief Financial Officer, concluded that as of December 31, 2013, our internal control over financial reporting was effective.\n(Note: The text is extracted from the given URL, which is a 10-K filing for Facebook, Inc. for the year ended December 31, 2013)\n',
     'Financial Statements and Supplementary Data': "The 'Financial Statements and Supplementary Data' section from the 10-K filing is:\nFinancial Statements and Supplementary Data\nFinancial Statements\nReport of Independent Registered Public Accounting Firm\nConsolidated Balance Sheets\nDecember 31, 2013 and 2012\nConsolidated Statements of Income\nYears ended December 31, 2013, 2012, and 2011\nConsolidated Statements of Comprehensive Income\nYears ended December 31, 2013, 2012, and 2011\nConsolidated Statements of Cash Flows\nYears ended December 31, 2013, 2012, and 2011\nNotes to Consolidated Financial Statements\nSupplementary Data\nQuarterly Financial Data (Unaudited)\nYears ended December 31, 2013 and 2012\nLet me know if you'd like me to extract any specific information from this section!\n",
     'Executive Compensation': 'The \'Executive Compensation\' section of the 10-K filing is:\nExecutive Compensation\nThe material in this report is not "soliciting material," is not deemed "filed" with the Commission and is not to be incorporated by reference into any filing of the registrant under the Securities Act of 1933 or the Securities Exchange Act of 1934, whether made before or after the date hereof and irrespective of any general incorporation language in any such filing.\nCompensation Discussion and Analysis\nOur executive compensation program is designed to attract, retain and motivate our executives to achieve our business objectives and increase long-term stockholder value. Our program is based on a pay-for-performance philosophy, which means that a significant portion of our executives\' compensation is tied to our performance.\nThe following compensation elements are provided to our named executive officers (NEOs):\nBase salary\nBonus\nEquity awards (restricted stock units (RSUs) and stock options)\nBenefits (health insurance, life insurance and disability insurance)\nOther benefits (such as use of company aircraft and security services)\nThe Compensation Committee of our Board of Directors (the Compensation Committee) is responsible for determining the compensation of our NEOs. The Compensation Committee works with its independent compensation consultant, Exequity LLP, to determine the appropriate compensation levels and mix of compensation elements for our NEOs.\nFor more information, see "Compensation Committee Report" and "Executive Compensation Tables" below.\nCompensation Committee Report\nThe Compensation Committee has reviewed and discussed the Compensation Discussion and Analysis with management, and based on such review and discussion, the Compensation Committee recommended to the Board of Directors that the Compensation Discussion and Analysis be included in this report.\nCompensation Committee\nMarc L. Andreessen\nErskine B. Bowles\nKenneth I. Chenault\nSusan D. Desmond-Hellmann\nMark J. Pincus\nExecutive Compensation Tables\nThe following tables set forth the compensation paid to our NEOs for the fiscal years ended December 31, 2013 and 2012.\nSummary Compensation Table\nName and Principal PositionYearSalaryBonusStock AwardsOption AwardsNon-Equity Incentive Plan CompensationChange in Pension Value and Nonqualified Deferred Compensation EarningsAll Other CompensationTotalMark Zuckerberg2013$1$266,859$3,989,965$$$$1,278,846$6,536,6702012$1$266,859$2,279,973$$$$1,016,167$3,563,999Sheryl K. Sandberg2013$333,333$$13,951,155$$$$1,278,846$15,563,3342012$321,927$$6,878,993$$$$1,016,167$8,216,087Michael W. Schroepfer2013$263,333$$2,599,996$$$$651,155$3,514,4842012$249,167$$1,599,996$$$$543,182$2,392,345David M. Wehner2013$175,000$$1,399,996$$$$431,155$2,005,1512012$150,000$$1,099,996$$$$343,182$1,592,178Christopher Cox2013$175,000$$1,399,996$$$$431,155$2,005,1512012$150,000$$1,099,996$$$$343,182$1,592,178\nLet me know if you have any further questions!\n',
     'Directors, Executive Officers and Corporate Governance': 'Here is the content of the "Directors, Executive Officers and Corporate Governance" section from the 10-K filing:\nDirectors, Executive Officers and Corporate Governance\nDirectors\nOur board of directors consists of eight members, including:\nMark Zuckerberg — Chairman of the Board of Directors and Chief Executive Officer\nSheryl Sandberg — Chief Operating Officer and Director\nMarc L. Andreessen — Director\nErskine B. Bowles — Director\nJames W. Breyer — Director\nDonald E. Graham — Director\nReed Hastings — Director\nPeter A. Thiel — Director\nExecutive Officers\nThe following individuals serve as our executive officers:\nMark Zuckerberg — Chairman of the Board of Directors and Chief Executive Officer\nSheryl Sandberg — Chief Operating Officer\nDavid M. Wehner — Chief Financial Officer\nMike Schroepfer — Chief Technology Officer\nChristopher Cox — Chief Product Officer\nCorporate Governance\nOur board of directors has established an audit committee, a compensation committee, and a nominating and governance committee. Our audit committee consists of Messrs. Breyer, Graham, and Thiel. Our compensation committee consists of Messrs. Andreessen, Bowles, and Hastings. Our nominating and governance committee consists of Messrs. Andreessen, Breyer, and Sandberg.\nLet me know if you have any further questions!\n',
     'Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters': "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters\nThe following table sets forth information regarding the beneficial ownership of our common stock as of December 31, 2013, by:\neach person or group of persons known by us to beneficially own more than 5% of our outstanding common stock;\neach of our directors;\neach of our executive officers; and\nall of our directors and executive officers as a group.\nName of Beneficial OwnerNumber of SharesPercentage of Outstanding SharesThe Vanguard Group, Inc.42,347,4125.3%BlackRock, Inc.34,762,1144.4%State Street Corporation29,704,6613.8%FMR, LLC (Fidelity)23,517,1123.0%Mark Zuckerberg426,274,79453.3%Sheryl K. Sandberg4,477,0220.6%David Wehner1,294,6540.2%Mike Schroepfer1,253,3340.2%David Ebersman1,242,1140.2%Donald E. Graham1,104,4260.1%Robert K. Kagle1,043,6690.1%Reed Hastings945,5620.1%Erskine B. Bowles844,1630.1%Susan D. Desmond-Hellmann744,1630.1%All directors and executive officers as a group (14 persons)433,342,52854.4%\nNote: The percentages are based on 793,655,205 shares of common stock outstanding as of December 31, 2013.\nPlease note that this information may not be up to date, and the filing is from 2013. If you need more recent information, please let me know, and I'll do my best to assist you.\n",
     'Management’s Discussion and Analysis of Financial Condition and Results of Operations': 'The \'Management’s Discussion and Analysis of Financial Condition and Results of Operations\' section from the given 10-K filing is reproduced below:\nManagement\'s Discussion and Analysis of Financial Condition and Results of Operations\nOur management\'s discussion and analysis of financial condition and results of operations ("MD&A") is provided in addition to, and should be read in conjunction with, our consolidated financial statements and related notes. Our MD&A includes the following sections:\nOverview: A summary of our business and significant trends and events that affect our financial condition and results of operations.\nResults of Operations: An analysis of our financial performance for the year ended December 31, 2013, compared to the year ended December 31, 2012.\nFinancial Condition: A discussion of our financial position as of December 31, 2013, compared to December 31, 2012.\nLiquidity and Capital Resources: A discussion of our liquidity and capital resources as of December 31, 2013.\nOff-Balance-Sheet Arrangements: A discussion of our off-balance-sheet arrangements as of December 31, 2013.\nContractual Obligations: A discussion of our contractual obligations as of December 31, 2013.\nCritical Accounting Standards and Estimates: A discussion of our critical accounting standards and estimates.\nRecently Adopted Accounting Pronouncements: A discussion of recently adopted accounting pronouncements.\nRecent Accounting Pronouncements Not Yet Adopted: A discussion of recent accounting pronouncements not yet adopted.\nPlease note that I\'ve extracted the content of this section as per your request. However, I recommend reading the entire 10-K filing to get a comprehensive understanding of the company\'s financial condition and results of operations.\nLet me know if you have any further questions or if there\'s anything else I can assist you with!\n',
     'Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities': "The requested text is quite long, but I've extracted the relevant section for you:\nMarket for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities\nOur Class A common stock is listed on the NASDAQ Global Select Market under the ticker symbol “FB”. The following table sets forth the high and low sales prices of our Class A common stock as reported on the NASDAQ Global Select Market for each quarter in the years ended December 31, 2013 and 2012:\nQuarterHighLow2013 QuartersFirst Quarter31.4522.67Second Quarter39.2924.88Third Quarter49.6636.87Fourth Quarter57.9745.182012 QuartersFirst Quarter43.9529.52Second Quarter41.3425.87Third Quarter33.0517.55Fourth Quarter31.4518.87\nAs of February 14, 2014, there were 4,299 holders of record of our Class A common stock and 1,043 holders of record of our Class B common stock.\nStock Performance Graph\nThe following graph shows the total return of an investment in our Class A common stock, the NASDAQ Composite Index, and the S&P 500 Index for the period from January 1, 2010 through December 31, 2013.\n[Insert Graphic]\nIssuer Purchases of Equity Securities\nWe did not purchase any of our equity securities during the year ended December 31, 2013.\nLet me know if you'd like me to extract any specific data points from this section!\n",
     'Cybersecurity': 'The text of the 10-K filing is not directly accessible to me. However, I can suggest ways for you to search the content yourself.\nYou can copy and paste the text into a word processor or text editor and search for "Cybersecurity". Alternatively, you can use the Find function (Ctrl + F) in your web browser to search the text.\nIf you provide the text, I can assist you in finding the specific section or content related to Cybersecurity.\n',
     'Selected Financial Data': "Selected Financial Data\nWe have audited the accompanying consolidated balance sheets of Facebook, Inc. as of December 31, 2013 and 2012, and the related consolidated statements of income, comprehensive income, and cash flows for each of the three years in the period ended December 31, 2013. Our audits also included the financial statement schedule listed in the Index at Item 15. We did not audit the financial statements of Facebook's subsidiaries, which statements reflect total assets of $5,443,215 and $3,534,292 as of December 31, 2013 and 2012, respectively, and total revenues of $3,771,572 and $2,415,306 for the years ended December 31, 2013 and 2012, respectively.\nSelected Financial Data\nThe following selected financial data is derived from our audited consolidated financial statements.\nYears Ended December 31,\n2013   * 2012   * 2011   * 2010   * 2009\n----   * ----   * ----   * ----   * ----\nRevenue   * $7,872   * $5,089   * $3,711   * $2,000   * $777\nNet income (loss)   *  $1,000   * $(696)   * $(564)\nNet income (loss) per share - diluted   *  $0.46   * $(0.29)   * $(0.29)\nTotal assets   * $18,435   * $9,446   * $6,313   * $3,446   * $1,632\nLong-term debt   * $1,500   * $ -   * $ -   * $ -   * $ -\nLet me know if you have any further requests!\n",
     'Competition': 'The content of the "Competition" item from the 10-K filing is:\nCompetition\nWe compete in the global online advertising and technology industries, which are highly competitive and rapidly evolving. We face significant competition in all areas of our business, including:\nOnline advertising: We compete with other companies that offer online advertising services, including Google, Microsoft, and Yahoo!.\nUser engagement: We compete with other companies that offer online services and products that compete for user attention and engagement, including other social media platforms, online search engines, and online entertainment providers.\nMobile: We compete with other companies that offer mobile apps and services that compete for user attention and engagement, including other social media platforms, mobile operating systems, and mobile apps.\nE-commerce: We compete with other companies that offer e-commerce services and products that compete for user attention and engagement, including other e-commerce platforms, online marketplaces, and brick-and-mortar retailers.\nSource: [1]\nPlease note that the text is extracted from a specific section of the filing and might not be comprehensive.\n',
     'Certain Relationships and Related Transactions, and Director Independence': 'The content of the "Certain Relationships and Related Transactions, and Director Independence" item from the 10-K filing is:\nCertain Relationships and Related Transactions, and Director Independence\nOur code of conduct and our Board of Directors are responsible for reviewing and approving transactions between us and any related parties. A related party is any person or entity that has a relationship with us that could influence the terms or manner of a transaction. Related parties include our directors, executive officers, nominees for election as directors, and their immediate family members.\nDuring 2013, there were no transactions with related parties that are required to be disclosed under the rules and regulations of the Securities and Exchange Commission.\nOur Board of Directors has determined that all of our directors, other than Mark Zuckerberg, are independent directors, as defined in the rules of the NASDAQ Stock Market. Our independent directors are: Marc Andreessen, Erskine Bowles, Kenneth Chenault, Susan Desmond-Hellmann, Donald Graham, Reed Hastings, Peter Thiel, and Jeffrey Zients.\nNote: The content has been extracted from the specified section of the 10-K filing and is presented in its original format, with minimal formatting adjustments for readability.\nPlease let me know if you have any further questions or requests!\n',
     'Changes in and Disagreements with Accountants on Accounting and Financial Disclosure': "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure\nNone.\nNote: The text of the 10-K filing does not contain the specified item. According to the SEC's guidelines, this item is only required if there are any changes in accountants or disagreements with accountants, which is not the case in this filing.\n",
     'Disclosure Regarding Foreign Jurisdictions that Prevent Inspections': 'Disclosure Regarding Foreign Jurisdictions that Prevent Inspections\nThe following is the content of the item:\n"We are subject to the inspection requirements of the Public Company Accounting Oversight Board (PCAOB). Our independent registered public accounting firm, Ernst & Young LLP, is subject to the PCAOB\'s inspection requirements. To our knowledge, there are no legal or regulatory proceedings or settlements that would prevent our independent registered public accounting firm from being subject to the PCAOB\'s inspection requirements.\nWe have been informed by our legal counsel in China that, based on the advice of the China Securities Regulatory Commission, the Chinese affiliates of the major accounting firms, including our independent registered public accounting firm, are not permitted to disclose to the PCAOB or any other foreign regulatory authority certain information regarding their audit work papers and other related documents. This could limit the PCAOB\'s ability to conduct inspections of our independent registered public accounting firm and could result in the PCAOB determining that our independent registered public accounting firm is not in compliance with the PCAOB\'s inspection requirements.\nIf the PCAOB determines that our independent registered public accounting firm is not in compliance with its inspection requirements, we could be subject to sanctions, including the inability to list our securities on a national securities exchange, which could adversely affect our business and the market price of our common stock."\nPlease note that the content is extracted from the provided URL and may not be up-to-date. The latest filings should be checked for the most recent information.\n',
     'Principal Accountant Fees and Services': 'Principal Accountant Fees and Services\nThe following table presents the fees billed by Ernst & Young LLP, our independent auditor, for the years ended December 31, 2013 and 2012:\nFees20132012Audit Fees$2,022,000$1,925,000Audit-Related Fees$30,000$30,000Tax Fees$135,000$130,000All Other Fees$0$0Total$2,187,000$2,085,000\nAudit Fees: These fees relate to the audit of our annual financial statements and the review of our quarterly financial statements.\nAudit-Related Fees: These fees relate to assurance and related services that are reasonably related to the performance of the audit or review of our financial statements.\nTax Fees: These fees relate to tax compliance, tax advice, and tax planning services.\nAll Other Fees: These fees relate to services other than those described above.\nSource: [Facebook, Inc. Form 10-K for the fiscal year ended December 31, 2013]((link unavailable))\n',
     'Exhibits and Financial Statement Schedules': "Here is the content of the 'Exhibits and Financial Statement Schedules' item from the 10-K filing:\nExhibits and Financial Statement Schedules\n(a) Exhibits\nExhibit 2.1: Agreement and Plan of Merger, dated as of July 1, 2012, by and among Facebook, Inc., Facebook Merger Sub, Inc., Instagram, Inc., and Shareholders’ Representative (incorporated by reference to Exhibit 2.1 to Facebook’s Current Report on Form 8-K filed on April 23, 2012)\nExhibit 3.1: Amended and Restated Certificate of Incorporation of Facebook, Inc. (incorporated by reference to Exhibit 3.1 to Facebook’s Quarterly Report on Form 10-Q filed on July 31, 2013)\nExhibit 3.2: Amended and Restated Bylaws of Facebook, Inc. (incorporated by reference to Exhibit 3.2 to Facebook’s Quarterly Report on Form 10-Q filed on July 31, 2013)\nExhibit 4.1: Form of Common Stock Certificate (incorporated by reference to Exhibit 4.1 to Facebook’s Registration Statement on Form S-1 filed on February 1, 2012)\nExhibit 10.2: 2012 Director Stock Plan (incorporated by reference to Exhibit 10.2 to Facebook’s Registration Statement on Form S-1 filed on February 1, 2012)\nExhibit 10.3: 2012 Director Stock Plan, Form of Stock Option Agreement (incorporated by reference to Exhibit 10.3 to Facebook’s Registration Statement on Form S-1 filed on February 1, 2012)\nExhibit 10.4: 2012 Director Stock Plan, Form of Restricted Stock Unit Agreement (incorporated by reference to Exhibit 10.4 to Facebook’s Registration Statement on Form S-1 filed on February 1, 2012)\nExhibit 10.5: Facebook, Inc. 2013 Compensation and Talent Acquisition Plan (incorporated by reference to Exhibit 10.5 to Facebook’s Quarterly Report on Form 10-Q filed on July 31, 2013)\nExhibit 10.6: Facebook, Inc. 2013 Equity Incentive Plan (incorporated by reference to Exhibit 10.6 to Facebook’s Quarterly Report on Form 10-Q filed on July 31, 2013)\nExhibit 10.7: Facebook, Inc. 2013 Equity Incentive Plan, Form of Stock Option Agreement (incorporated by reference to Exhibit 10.7 to Facebook’s Quarterly Report on Form 10-Q filed on July 31, 2013)\nExhibit 10.8: Facebook, Inc. 2013 Equity Incentive Plan, Form of Restricted Stock Unit Agreement (incorporated by reference to Exhibit 10.8 to Facebook’s Quarterly Report on Form 10-Q filed on July 31, 2013)\nExhibit 21.1: Subsidiaries of Facebook, Inc.\nExhibit 23.1: Consent of Ernst & Young LLP, Independent Registered Public Accounting Firm\nExhibit 31.1: Certification of Chief Executive Officer pursuant to Rule 13a-14(a) or Rule 15d-14(a) under the Securities Exchange Act of 1934\nExhibit 31.2: Certification of Chief Financial Officer pursuant to Rule 13a-14(a) or Rule 15d-14(a) under the Securities Exchange Act of 1934\nExhibit 32.1: Certification of Chief Executive Officer pursuant to Rule 13a-14(b) or Rule 15d-14(b) under the Securities Exchange Act of 1934 and 18 U.S.C. Section 1350\nExhibit 32.2: Certification of Chief Financial Officer pursuant to Rule 13a-14(b) or Rule 15d-14(b) under the Securities Exchange Act of 1934 and 18 U.S.C. Section 1350\nExhibit 101.INS: XBRL Instance Document\nExhibit 101.SCH: XBRL Taxonomy Extension Schema Document\nExhibit 101.CAL: XBRL Taxonomy Extension Calculation Linkbase Document\nExhibit 101.LAB: XBRL Taxonomy Extension Label Linkbase Document\nExhibit 101.PRE: XBRL Taxonomy Extension Presentation Linkbase Document\nExhibit 101.DEF: XBRL Taxonomy Extension Definition Linkbase Document\n(b) Financial Statement Schedules\nSchedule II: Valuation and Qualifying Accounts\nSchedule III: Real Estate Investments\nLet me know if you have any further requests!\n"}



Testing message generation with Meta.ai API


```python
ai.prompt(message="""
Use the the most prominent aspect of the below a reciever's information to generate a short first line that includes a greeting and affirmation/congratulation/compliment of their experience to be used for an intial linkedin personalized cold outreach message that hooks their attention, reply as if you are sending me that message, so only return the sentence itself and nothing else:

ITEM 1. BUSINESS
Proposed Merger with General Mills, Inc.
On February 22, 2018, Blue Buffalo Pet Products, Inc., General Mills, Inc., a Delaware corporation (“GMI”), and Bravo Merger Corp., a Delaware corporation and wholly-owned subsidiary of GMI (“Merger Sub”), entered into an Agreement and Plan of Merger (the “Merger Agreement”), pursuant to which, on the terms and subject to the conditions set forth in the Merger Agreement, GMI agreed to acquire the Company for $40.00 per common share, in cash.
On the terms and subject to the conditions set forth in the Merger Agreement, Merger Sub will merge with and into the Registrant (the “Merger”), with the Registrant continuing as the surviving company and as a wholly-owned subsidiary of GMI, and our common stock will be delisted from the NASDAQ Stock Market. At the effective time of the Merger, each share of our common stock (other than shares of the Company’s common stock held by GMI, Merger Sub or any other wholly-owned subsidiary of GMI, shares owned by the Company (including shares held in treasury) or any of its wholly-owned subsidiaries, and shares owned by stockholders who have properly exercised and perfected appraisal rights under Delaware law) will be converted into the right to receive $40.00 in cash per share.
The boards of directors of each of GMI and the Registrant have approved the Merger Agreement and holders of more than 50% of the Registrant’s outstanding shares have approved the transaction. While no other approval of the Registrant’s Board of Directors or shareholders is required to complete the transaction, the closing of the Merger is subject to certain regulatory approvals and other customary closing conditions.
There is no assurance that the Merger will occur. See “Risk Factors-Risks Related to the Pending Merger” in “Part I-Item 1A. Risk Factors.”
Overview
We are one of the fastest growing major pet food companies in the United States, selling dog and cat food made with whole meats, fruits and vegetables, and other high-quality, natural ingredients. BLUE is a billion dollar brand and is the #1 brand in the Wholesome Natural market segment. We currently have approximately 7% share of the U.S. pet food market and feed only 3% of the 184 million pets in the United States. With a proven new user acquisition strategy, we are committed to converting more pet parents into True Blue Believers and continuing to increase our share of the attractive $28 billion U.S. pet food market.
We believe we have built an exceptional company with a breakthrough brand and an innovative business model, backed by our mission and our belief in a large unmet consumer demand for pet food with high-quality, natural ingredients.
Our History
Today, Blue Buffalo Pet Products, Inc. (“BBPP”, and together with its subsidiaries, the “Company,” “we,” “us,” “its,” and “our”) conducts its business exclusively through its wholly-owned operating
""")['message']
```




    'Hi, I wanted to reach out and congratulate you on the impressive merger with General Mills, Inc. - quite an achievement!\n'




```python
ai.prompt(message="""
https://www.sec.gov/Archives/edgar/data/1694426/000169442624000028/dk-20231231.htm#i6fa8754773d643bcbfbf1e33d05e1689_265
Read the above 10-K filing link, if there isn't a item called 'MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES', return 'None" and nothing else; if there is, provide the above link's content of item 'MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES'.
""")['message']
```




    'MARKET FOR REGISTRANT\'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES\nAs of December 31, 2023, the Company\'s common stock is listed on the New York Stock Exchange (NYSE) under the ticker symbol "DK".\nThe following table sets forth the high and low sales prices for the Company\'s common stock as reported on the NYSE for the periods indicated:\nQuarterHighLowFirst Quarter 2023$63.95$45.06Second Quarter 2023$73.94$56.12Third Quarter 2023$81.11$65.23Fourth Quarter 2023$95.00$74.51\nAs of December 31, 2023, there were approximately 342 holders of record of the Company\'s common stock.\nThe Company did not repurchase any shares of its common stock during the year ended December 31, 2023.\nThe Company\'s board of directors has authorized a total of $10.0 billion for share repurchases under its share repurchase program, of which $5.3 billion remained available as of December 31, 2023.\n'



Packaged Function


```python
def get_10k(company_name):
    cik_url = get_cik_url(company_name)
    html_url = get_10k_url(cik_url)
    html_content = get_10k_file(html_url)
    result = extract_sections(html_content)
    return result
```
