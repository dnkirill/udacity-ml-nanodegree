# Udacity Machine Learning Nanodegree Projects
This is a collection of projects required to complete Udacity ML nanodegree. This collection is already outdated though: folks at Udacity are doing a good job by constantly adding more material to their Nanodegrees.

There are four projects ranging from a _very_ simple Project 1 to a considerably more involved Project 3 and Project 4. Each project (except P1) focuses on a specific domain of ML: supervised, unsupervised and reinforcement learning. Looking backwards, I find all the projects (especially P1 and P2) quite easy, but there should be something to start from!

* [**Project 1: Predicting Boston Housing Prices**](https://github.com/dnkirill/udacity-ml-nanodegree/blob/master/p1_boston_houses_prices_project/boston_houses_prices.ipynb). A very simple project to get acquainted to working with basic machine learning techniques using numpy and scipy. It relies on the famous Boston houses prices dataset which can be accessed directly from sklearn. Regression problem, supervised learning.

* [**Project 2: Building a Student Intervention System**](https://github.com/dnkirill/udacity-ml-nanodegree/blob/master/p2_supervised_learning/student_intervention.ipynb). A supervised learning project. The goal of this project is to identify students who might need an early intervention based on their exams' scores. 

  We compare the the performance of three models: SVM, Logistic Regression and Naive Bayes based on their F1 score, minimum required training set and training time, choose the best of them, tune it a little bit with Grid Search and, finally, calculate its ROC AUC score.

* [**Project 3: Creating Customer Segments**](https://github.com/dnkirill/udacity-ml-nanodegree/blob/master/p3_unsupervised_learning/customer_segments.ipynb). An unsupervised learning project. We analyze a [Wholesale customers dataset](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) from UCI. The goal is to identify customer segments in data and to propose a personalized product delivery schedule for each customer segment. This project is more complicated than the previous two. Highlights:

  * After visualizing distributions of each feature, we log-transform them to make their distributions closer to normal. We detect and remove outliers using Tukey's method  based on IQR.
  * We apply PCA transformation and try to analyze and visualize the resulted components. 
  * Finally, we pick K-Means over GMM for clustering, choose the best cluster number based on silhouette score and analyze what customer segments these clusters may represent. 
  * In the end, we discuss how to run A/B tests with the information we've just got to make sure that the modifications in delivery schedules don't negatively affect large groups of our customers. We also discuss how to measure the significance of A/B tests in the real world.

* [**Project 4. Training a Smartcab to Drive**](https://github.com/dnkirill/udacity-ml-nanodegree/blob/master/p4_reinforcement_learning/project4_report.ipynb). In this project we train a smart agent to drive a simple virtual town. We need to consider other cars on the road, traffic lights and basic turning rules and to make our smart agent reach the destination in time. The project requires `pygame` GUI programming module with not-that-obvious setup process. I suggest installing it with `anaconda` enviroment (though I don't use it on a regular basis). Highlights:

  * We use Markov Decision Process (MDP) and, specifically, Q-Learning, to describe possible actions for each state the smart agent finds itself in. We stick to a simple Q-Table-based algorithm without DQN or policy gradients and research different behaviors of our smart agents.

  * We try to optimize the learning by shrinking down the Q-Table size and tuning hyperparameters.

  * Finally, we compare two algorithm for action selection: GLIE (greedy learning + infinite exploration) and softmax action selection. 

  Personally, this has been the most challenging and informative project of all of these four. Most of the work is made in python files and the jupyter notebook just reports the results.

  N.B. I decided not to add logs from this project (large files) so running the code in the notebook will lead to blank plots — keep this in mind.