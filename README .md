# TruthRev: Identifying Fake vs Genuine Reviews  

## 📌 Problem Statement  
Online reviews play a crucial role in consumer decision-making. However, fake reviews distort opinions and reduce trust in online platforms. This project aims to detect whether a review is genuine (human-written) or fake (computer-generated), thereby helping both platforms and users make better-informed decisions.  

---

## 📊 Dataset Overview  
The dataset contains thousands of reviews across multiple product categories. Each review includes:  
- **Review Text:** Content of the review  
- **Rating:** Score given by the reviewer  
- **Label:**  
  - `OR` → Original Review (Genuine)  
  - `CG` → Computer-Generated (Fake)  

---

## 🛠️ Libraries and Tools  
This project uses the following Python libraries:  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Natural Language Processing:** NLTK, string, nltk.corpus  
- **Machine Learning:** scikit-learn (Logistic Regression, SVM, Decision Trees, Random Forest, KNN, Naive Bayes)   

---

## 📝 Text Preprocessing  
The following preprocessing steps were applied to the reviews:  
1. Remove punctuation and numbers  
2. Convert text to lowercase  
3. Remove stopwords  
4. Apply stemming and lemmatization  

---

## 🔎 Feature Extraction  
Text was converted into numerical features using:  
- **Bag of Words (BoW):** CountVectorizer  
- **TF-IDF:** TfidfVectorizer  

---

## 🤖 Machine Learning Models  
The following models were implemented:  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machines (SVM)  
- Multinomial Naive Bayes  

---

## 📈 Model Performance  
| Model                        | Accuracy (%) |
|-------------------------------|--------------|
| Support Vector Machines (SVM) | **88.08**    |
| Logistic Regression           | 86.42        |
| Multinomial Naive Bayes       | 84.73        |
| Random Forest Classifier      | 84.25        |
| Decision Tree Classifier      | 74.02        |
| K-Nearest Neighbors (KNN)     | 57.93        |

---

## ✅ Key Observation  
Support Vector Machines (SVM) achieved the **highest accuracy (88.08%)**, making it the most effective model for detecting fake reviews in this project.  

