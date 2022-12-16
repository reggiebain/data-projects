# data-projects
### Guide to Reggie Bain's Data Projects Repo
#### 1. Predicting Delivery Times
  - Compares a variety of classical ML algorithms performance at predicting DoorDash delivery times using a publically available data set.
  - Ultimately, XGBoost is found to be the best performing model, overall predicting delivery times to within a reasonable +/-.
  - Utlilizes several key techniques for feature selection and feature engineering, including Principal Component Analysis (PCA)
  - and a Gaussian Mixture Model (GMM) to identify the most important features.
#### 2. Sentiment Flair Analysis + Clustering Analysis
  - Uses Flair, a state-of-the-art pre-trained model for NLP, to analyze the sentiment of unlabeled customer comments on pharmacudical products.
  - Leverages regex and other techniques to clean reviews, which come from a variety of sources including social media and surveys
  - Use clustering with Google Word2Vec package to create embeddings for reviews to analyze sentiment in the same customer reviews.
#### 3. Rotten Tomatoes Movie Ratings Prediction
  - Based on audience rating data, uses XGBoost and compares with other classical ML techniques to predict whether movies are Fresh, Certified Fresh,
  - or Rotten. Works with combination of numerical, categorical, and ordinal data. Uses one-hot and ordinal encodings, feature scaling, and other data wrangling techniques to regularize features. 
  
