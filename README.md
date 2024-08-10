# Modeling-Car-Insurance-Claim-Outcomes
Clean customer data and use logistic regression to predict whether people will make a claim on their car insurance!

# Description: 
Machine learning is very popular in the insurance market, which has long been renowned for being data-driven.

Working on behalf of On the Road car insurance, you'll investigate their customer data to identify the feature that produces the most accurate logistic regression model for them to predict whether a claim will be made against a policy!

# Project Instructions
Identify the single feature of the data that is the best predictor of whether a customer will put in a claim (the "outcome" column), excluding the "id" column.
Store as a DataFrame called best_feature_df, containing columns named "best_feature" and "best_accuracy" with the name of the feature with the highest accuracy, and the respective accuracy score.

![image](https://github.com/user-attachments/assets/34c98ef8-3835-44dd-96af-70a0c108a8be)


Insurance companies invest a lot of [time and money](https://www.accenture.com/_acnmedia/pdf-84/accenture-machine-leaning-insurance.pdf) into optimizing their pricing and accurately estimating the likelihood that customers will make a claim. In many countries insurance it is a legal requirement to have car insurance in order to drive a vehicle on public roads, so the market is very large!

Knowing all of this, On the Road car insurance have requested your services in building a model to predict whether a customer will make a claim on their insurance during the policy period. As they have very little expertise and infrastructure for deploying and monitoring machine learning models, they've asked you to identify the single feature that results in the best performing model, as measured by accuracy, so they can start with a simple model in production.

They have supplied you with their customer data as a csv file called `car_insurance.csv`, along with a table detailing the column names and descriptions below.
