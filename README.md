# data-projects
### Guide to Reggie Bain's Data Projects Repo
#### 1. NLI with HuggingFace+Tensorflow and Custom Classification Head (xlm-roberta-nli-task.ipynb)
  - Achieves 92.7% test accuracy on Kaggle Contradictory, My Dear Watson dataset leveraging Kaggle TPU's.
  - Fine-tunes a XLM-RoBERTa-XNLI pre-trained model from HuggingFace, adding a custom classification head to the model and performing fine tuning.
  - Leverage Tensorflow, Keras, and Transformers APIs to analyze pairs of sentences and classify the pairs as having entailment, nuetral, or contradictory meanings.
#### 2. Predicting Delivery Times
  - Compares a variety of classical ML algorithms performance at predicting DoorDash delivery times using a publically available data set.
  - Ultimately, XGBoost is found to be the best performing model, overall predicting delivery times to within a reasonable +/-.
  - Utlilizes several key techniques for feature selection and feature engineering, including Principal Component Analysis (PCA)
  - and a Gaussian Mixture Model (GMM) to identify the most important features.
#### 3. Movie Review Sentence Classifier (Deep Learning)
  - Built an LSTM model using Keras and Tensorflow to classify the sentiment of critic reviews of movies using Rotten Tomatoes data. 
  - Additionally, worked on fine-tuning BERT to predict the same critic review sentiments. Leverage the HuggingFace Transformers library as well as Keras/Tensorflow functionality.
#### 4. Sentiment Flair Analysis + Clustering Analysis
  - Uses Flair, a state-of-the-art pre-trained model for NLP, to analyze the sentiment of unlabeled customer comments on pharmacudical products.
  - Leverages regex and other techniques to clean reviews, which come from a variety of sources including social media and surveys
  - Use clustering with Google Word2Vec package to create embeddings for reviews to analyze sentiment in the same customer reviews.
#### 5. Rotten Tomatoes Movie Ratings Prediction
  - Based on audience rating data, uses XGBoost and compares with other classical ML techniques to predict whether movies are Fresh, Certified Fresh,
  - or Rotten. Works with combination of numerical, categorical, and ordinal data. Uses one-hot and ordinal encodings, feature scaling, and other data wrangling techniques to regularize features. 
#### 6. Predicting Play Calls
  - Leverages a novel API that accesses College Football Data to create a model of play calling for SEC teams. 
  - Uses ensemble methods to help identify the most important features to predict play calls by coaches depending on the time left in the game, down, distance, and other key football-specific metrics.
#### 7. Football ELO Ratings
  - Leverage novel college football data API to create power ranking system using ELO ratings, the system made famous for ranking chess players.
  - Develops ELO ratings over time between 2000-Present by taking information from all games played by all teams.
#### 8. Sarcasm in News Headlines
  - Build ANN using TensorFlow to detect sarcasm in news headlines using Kaggle dataset.
#### 9. Forecasting Store Sales
  - Use Neural Prophet (Facebook) to forecast sales at stores in Ecuador using Kaggle dataset. Break down trend, seasonality, and incorporate key holidays and additional regression (nation-wide oil prices). Tune hyperparameters using parameter grid search.
#### 10. Chat with a Database
  - Using Langchain and a Pinecone vector store, I create a small database of documents including a chapter from an open source physics textbook and a syllabus for a real course. I create embeddings and use Retrieval Augmented Generation (RAG) to leverage both open source (Falcon 7B-Instruct) and non open source (OpenAI GPT 3.5) LLMs to query the database with excellent results!
#### 11. College Enrollment EDA
  - Using various data manipulation and visualization techniques, I analyze enrollment data from colleges and universities around the world. In a data driven way, I show and discuss significant historial events and their coincidence with major changes in enrollments.
