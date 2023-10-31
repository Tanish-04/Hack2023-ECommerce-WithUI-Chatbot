![image](https://github.com/Tanish-04/Hack2023-ECommerce-WithUI-Chatbot/assets/63210891/41ffecea-6025-4e5f-9d2b-8db283d705ce)<div align="center">
  <a href="https://wm2762smemmk2ypdmmxtnb.streamlit.app/">
  </a>

  <div id="user-content-toc">
    <ul>
      <summary><h1 style="display: inline-block;">Kamal Chatbot</h1></summary>
    </ul>
  </div>
  
  <p>A Customer Service Chatbot Trained on Decathlon Maroc Data</p>
<be>

<details>
<summary>Table of Contents</summary>

- [Project Overview](#introduction)
- [Sentiment Analysis & Chatbot](#parts)
- [References](#refs)
- [Technologies](#techs)
- [Installation](#installation)
- [Contact](#contact)

</details>

<hr>

### 💬 Chatbot Page
![image](https://github.com/Tanish-04/Hack2023-ECommerce-WithUI-Chatbot/assets/63210891/94a17501-0f93-45e0-9297-f02de17e16c3)

<a name="introduction"></a>
## 🔬 Project Overview :

### 🎯 Goal :

This project consists of a sentiment analysis part for product reviews and a chatbot that interacts with customers. The chatbot utilizes the GPT-3.5-turbo model from OpenAI to provide information and assistance to customers.

<a name="parts"></a>
## 🤖 Sentiment Analysis & Chatbot

### 1. 😍😐😡 Sentiment Analysis :

The sentiment analysis part focuses on analyzing product reviews. It follows the following steps:

1. **Scraping:** The reviews are scraped from the Amazon website using Beautiful Soup, extracting the necessary information for analysis like customer name, review date, review description, review rating, etc...
2. **Data Cleaning:** The scraped reviews are cleaned to remove any irrelevant or noisy data like texts or numeric values, ensuring that the analysis focuses on meaningful and valid content.

3. **Sentiment Analysis:** The Natural Language Toolkit (NLTK) library and Vader sentiment analysis are employed to assess the sentiment of the review descriptions. This process provides an understanding of the customers' opinions and sentiments towards specific products. 


### 2. 💬 Chatbot :

The chatbot is designed to engage with customers and provide them with relevant information and assistance regarding various aspects of Amazon operation. It follows the subsequent steps:

1. **Data Extraction**: HTML files containing information about returns policies, warranty details, contact information, and other relevant topics are loaded using a WebBaseloader in Langchain.

2. **Document Embedding**: The OpenAI Ada embedding model is utilized to convert the loaded documents into embeddings, enabling efficient search and retrieval of relevant information.

3. **Vector Database:** The embeddings generated from the documents are loaded into a Pinecone vector database. This database facilitates similarity searches, allowing the chatbot to retrieve the most relevant parts of the documents based on customer queries.

4. **Chatbot Interface:** A Streamlit application is developed with a chatbot interface, enabling direct interaction with customers. Customers can inquire about specific problems or request information, and the chatbot (gpt-3.5-model with custom template) leverages the Pinecone database to provide the most appropriate responses based on document similarity.
   

<a name="refs"></a>
## 📋 References

**Project inspired by**: https://www.youtube.com/watch?v=cVA1RPsGQcw  
**Beautiful Soup Scraper:** https://github.com/umangkejriwal1122/Web-Scraping  
**Streamlit Chatbot UI inspired by**: https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/

<a name="techs"></a>
## 🛠️ Technologies Used

![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
<img src="https://www.jeveuxetredatascientist.fr/wp-content/uploads/2022/06/BeautifulSoup-1080x428.jpg" alt="beautifulsoup" width="75">
![Openai](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)
<img src="https://www.datanami.com/wp-content/uploads/2022/03/pinecone_logo.png" alt="pinecone" width="80">
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
<img src="https://user-images.githubusercontent.com/66017329/223900076-e1d5c1e5-7c4d-4b73-84e7-ae7d66149bc6.png" alt="Banner" width="100">


<a name="installation"></a>
## 🖥️ Installation : 
1. Clone the repository:

```git clone https://github.com/Tanish-04/Hack2023-ECommerce-WithUI-Chatbot```

2. Install the required packages:

```pip install -r requirements.txt```

### Usage : 

1. Change directory to `Hack2023-ECommerce-WithUI-Chatbot`:

```cd Hack2023-ECommerce-WithUI-Chatbot```

2. Run the app:

```streamlit run app.py```


