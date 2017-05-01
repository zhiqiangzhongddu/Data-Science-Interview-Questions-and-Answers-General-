Data-Science-Interview-Questions-and-Answers-General
====================================================
I hope this article could help beginners to better understanding of Data Science, and have a better performance in your first interviews.  
I will do long update and please feel free to contact me if you have any questions.  
I'm just a porter, most of them are borrowing from others

## Data Science Questions and Answers (General) for beginner
### Editor : Zhiqiang ZHONG 

# Content
#### Q1 How would you create a taxonomy to identify key customer trends in unstructured data?
    The best way to approach this question is to mention that it is good to check with the business owner 
    and understand their objectives before categorizing the data. Having done this, it is always good to 
    follow an iterative approach by pulling new data samples and improving   the model accordingly by validating 
    it for accuracy by soliciting feedback from the stakeholders of the business. This helps ensure that your 
    model is producing actionable results and improving over the time.
#### Q2 Python or R – Which one would you prefer for text analytics?
    The best possible answer for this would be Python because it has Pandas library that provides easy to use 
    data structures and high performance data analysis tools.
#### Q3 Which technique is used to predict categorical responses?
    Classification technique is used widely in mining for classifying data sets.
#### Q4 What is logistic regression? Or State an example when you have used logistic regression recently.
    Logistic Regression often referred as logit model is a technique to predict the binary outcome from a linear 
    combination of predictor variables. For example, if you want to predict whether a particular political leader 
    will win the election or not. In this case, the outcome of prediction is binary i.e. 0 or 1 (Win/Lose). The 
    predictor variables here would be the amount of money spent for election campaigning of a particular candidate, 
    the amount of time spent in campaigning, etc.
#### Q5 What are Recommender Systems?
    A subclass of information filtering systems that are meant to predict the preferences or ratings that a user 
    would give to a product. Recommender systems are widely used in movies, news, research articles, products, 
    social tags, music, etc.
#### Q6 Why data cleaning plays a vital role in analysis?
    Cleaning data from multiple sources to transform it into a format that data analysts or data scientists can work 
    with is a cumbersome process because - as the number of data sources increases, the time take to clean the data 
    increases exponentially due to the number of sources and the volume of data generated in these sources. It might 
    take up to 80% of the time for just cleaning data making it a critical part of analysis task.
#### Q7 Differentiate between univariate, bivariate and multivariate analysis.
    These are descriptive statistical analysis techniques which can be differentiated based on the number of 
    variables involved at a given point of time. For example, the pie charts of sales based on territory involve 
    only one variable and can be referred to as univariate analysis.

    If the analysis attempts to understand the difference between 2 variables at time as in a scatterplot, then it 
    is referred to as bivariate analysis. For example, analysing the volume of sale and a spending can be considered 
    as an example of bivariate analysis.

    Analysis that deals with the study of more than two variables to understand the effect of variables on the 
    responses is referred to as multivariate analysis.

#### Q8 What do you understand by the term Normal Distribution?
    Data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled
    up. However, there are chances that data is distributed around a central value without any bias to the left or
    right and reaches normal distribution in the form of a bell shaped curve. The random variables are distributed
    in the form of an symmetrical bell shaped curve.
![](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Bell+Shaped+Curve+for+Normal+Distribution.jpg)
#### Q9 What is Linear Regression?
    Linear regression is a statistical technique where the score of a variable Y is predicted from the score of a 
    second variable X. X is referred to as the predictor variable and Y as the criterion variable.
#### Q10 What is Interpolation and Extrapolation?
    Estimating a value from 2 known values from a list of values is Interpolation. Extrapolation is approximating 
    a value by extending a known set of values or facts.
#### Q11 What is power analysis?
    An experimental design technique for determining the effect of a given sample size.
#### Q12 What is K-means? How can you select K for K-means?
#### Q13 What is Collaborative filtering?
    The process of filtering used by most of the recommender systems to find patterns or information by collaborating 
    viewpoints, various data sources and multiple agents.
#### Q14 What is the difference between Cluster and Systematic Sampling?
    Cluster sampling is a technique used when it becomes difficult to study the target population spread across
    a wide area and simple random sampling cannot be applied. Cluster Sample is a probability sample where each 
    sampling unit is a collection, or cluster of elements. Systematic sampling is a statistical technique where 
    elements are selected from an ordered sampling frame. In systematic sampling, the list is progressed in a 
    circular manner so once you reach the end of the list,it is progressed from the top again. The best example
    for systematic sampling is equal probability method.
#### Q15 Are expected value and mean value different?
    They are not different but the terms are used in different contexts. Mean is generally referred when talking 
    about a probability distribution or sample population whereas expected value is generally referred in a 
    random variable context.

    For Sampling Data
    Mean value is the only value that comes from the sampling data.
    Expected Value is the mean of all the means i.e. the value that is built from multiple samples. Expected 
    value is the population mean.

    For Distributions
    Mean value and Expected value are same irrespective of the distribution, under the condition that the 
    distribution is in the same population.
#### Q16 What does P-value signify about the statistical data?
    P-value is used to determine the significance of results after a hypothesis test in statistics. P-value 
    helps the readers to draw conclusions and is always between 0 and 1.
- P- Value > 0.05 denotes weak evidence against the null hypothesis which means the null hypothesis cannot be rejected.
- P-value <= 0.05 denotes strong evidence against the null hypothesis which means the null hypothesis can be rejected.
- P-value=0.05is the marginal value indicating it is possible to go either way.
#### Q17 Do gradient descent methods always converge to same point?
    No, they do not because in some cases it reaches a local minima or a local optima point. You don’t reach 
    the global optima point. It depends on the data and starting conditions
#### Q18 What are categorical variables?
#### Q19 A test has a true positive rate of 100% and false positive rate of 5%. There is a population with a 1/1000 rate of having the condition the test identifies. Considering a positive test, what is the probability of having that condition?
    Let’s suppose you are being tested for a disease, if you have the illness the test will end up saying you 
    have the illness. However, if you don’t have the illness- 5% of the times the test will end up saying you
    have the illness and 95% of the times the test will give accurate result that you don’t have the illness. 
    Thus there is a 5% error in case you do not have the illness.

    Out of 1000 people, 1 person who has the disease will get true positive result.

    Out of the remaining 999 people, 5% will also get true positive result.

    Close to 50 people will get a true positive result for the disease.

    This means that out of 1000 people, 51 people will be tested positive for the disease even though only one 
    person has the illness. There is only a 2% probability of you having the disease even if your reports say 
    that you have the disease.

#### Q20 How you can make data normal using Box-Cox transformation?
#### Q21 What is the difference between Supervised Learning an Unsupervised Learning?
    If an algorithm learns something from the training data so that the knowledge can be applied to the test data,
    then it is referred to as Supervised Learning. Classification is an example for Supervised Learning. If the
    algorithm does not learn anything beforehand because there is no response variable or any training data, 
    then it is referred to as unsupervised learning. Clustering is an example for unsupervised learning.
#### Q22 Explain the use of Combinatorics in data science.
#### Q23 Why is vectorization considered a powerful method for optimizing numerical code?
#### Q24 What is the goal of A/B Testing?
    It is a statistical hypothesis testing for randomized experiment with two variables A and B. The goal of A/B 
    Testing is to identify any changes to the web page to maximize or increase the outcome of an interest. An
    example for this could be identifying the click through rate for a banner ad.
#### Q25 What is an Eigenvalue and Eigenvector?
    Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the
    eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular
    linear transformation acts by flipping, compressing or stretching. Eigenvalue can be referred to as the strength
    of the transformation in the direction of eigenvector or the factor by which the compression occurs.
#### Q26 What is Gradient Descent?
#### Q27 How can outlier values be treated?
    Outlier values can be identified by using univariate or any other graphical analysis method. If the number of
    outlier values is few then they can be assessed individually but for large number of outliers the values can
    be substituted with either the 99th or the 1st percentile values. All extreme values are not outlier values.
    The most common ways to treat outlier values –
1. To change the value and bring in within a range
2. To just remove the value.
#### Q28 How can you assess a good logistic model?
    There are various methods to assess the results of a logistic regression analysis-
- Using Classification Matrix to look at the true negatives and false positives.
- Concordance that helps identify the ability of the logistic model to differentiate between the event happening and not happening.
- Lift helps assess the logistic model by comparing it with random selection.
#### Q29 What are various steps involved in an analytics project?
- Understand the business problem
- Explore the data and become familiar with it.
- Prepare the data for modelling by detecting outliers, treating missing values, transforming variables, etc.
- After data preparation, start running the model, analyse the result and tweak the approach. This is an iterative step till the best possible outcome is achieved.
- Validate the model using a new data set.
- Start implementing the model and track the result to analyse the performance of the model over the period of time.
#### Q30 How can you iterate over a list and also retrieve element indices at the same time?
    This can be done using the enumerate function which takes every element in a sequence just like in a list
    and adds its location just before it.
#### Q31 During analysis, how do you treat missing values?
    The extent of the missing values is identified after identifying the variables with missing values. If 
    any patterns are identified the analyst has to concentrate on them as it could lead to interesting and 
    meaningful business insights. If there are no patterns identified, then the missing values can be 
    substituted with mean or median values (imputation) or they can simply be ignored.There are various
    factors to be considered when answering this question-

- Understand the problem statement, understand the data and then give the answer.Assigning a default value which can be mean, minimum or maximum value. Getting into the data is important.
- If it is a categorical variable, the default value is assigned. The missing value is assigned a default value.
- If you have a distribution of data coming, for normal distribution give the mean value.
- Should we even treat missing values is another important point to consider? If 80% of the values for a variable are missing then you can answer that you would be dropping the variable instead of treating the missing values.
#### Q32 Explain about the box cox transformation in regression models.
#### Q33 Can you use machine learning for time series analysis?
    Yes, it can be used but it depends on the applications.
#### Q34 Write a function that takes in two sorted lists and outputs a sorted list that is their union. 
    First solution which will come to your mind is to merge two lists and short them afterwards
    Python code-
    def return_union(list_a, list_b):
        return sorted(list_a + list_b)
    
    R code-
    return_union <- function(list_a, list_b)
    {
    list_c<-list(c(unlist(list_a),unlist(list_b)))
    return(list(list_c[[1]][order(list_c[[1]])]))
    }

    Generally, the tricky part of the question is not to use any sorting or ordering function. In that 
    case you will have to write your own logic to answer the question and impress your interviewer.
    
    Python code-
    def return_union(list_a, list_b):
        len1 = len(list_a)
        len2 = len(list_b)
        final_sorted_list = []
        j = 0
        k = 0
    
        for i in range(len1+len2):
            if k == len1:
                final_sorted_list.extend(list_b[j:])
                break
            elif j == len2:
                final_sorted_list.extend(list_a[k:])
                break
            elif list_a[k] < list_b[j]:
                final_sorted_list.append(list_a[k])
                k += 1
            else:
                final_sorted_list.append(list_b[j])
                j += 1
        return final_sorted_list

    Similar function can be returned in R as well by following the similar steps.

    return_union <- function(list_a,list_b)
    {
    #Initializing length variables
    len_a <- length(list_a)
    len_b <- length(list_b)
    len <- len_a + len_b
    
    #initializing counter variables
    
    j=1
    k=1
    
    #Creating an empty list which has length equal to sum of both the lists
    
    list_c <- list(rep(NA,len))
    
    #Here goes our for loop 
    
    for(i in 1:len)
    {
        if(j>len_a)
        {
            list_c[i:len] <- list_b[k:len_b]
            break
        }
        else if(k>len_b)
        {
            list_c[i:len] <- list_a[j:len_a]
            break
        }
        else if(list_a[[j]] <= list_b[[k]])
        {
            list_c[[i]] <- list_a[[j]]
            j <- j+1
        }
        else if(list_a[[j]] > list_b[[k]])
        {
        list_c[[i]] <- list_b[[k]]
        k <- k+1
        }
    }
    return(list(unlist(list_c)))

    }
#### Q35 What is the difference between Bayesian Inference and Maximum Likelihood Estimation (MLE)?
#### Q36 What is Regularization and what kind of problems does regularization solve?
#### Q37 What is multicollinearity and how you can overcome it?
#### Q38 What is the curse of dimensionality?
#### Q39 How do you decide whether your linear regression model fits the data?
#### Q40 What is the difference between squared error and absolute error?
#### Q41 What is Machine Learning?
    The simplest way to answer this question is – we give the data and equation to the machine. Ask the
    machine to look at the data and identify the coefficient values in an equation.

    For example for the linear regression y=mx+c, we give the data for the variable x, y and the machine
    learns about the values of m and c from the data.
#### Q42 How are confidence intervals constructed and how will you interpret them?
#### Q43 How will you explain logistic regression to an economist, physican scientist and biologist?
#### Q44 How can you overcome Overfitting?
#### Q45 Differentiate between wide and tall data formats?
#### Q46 Is Naïve Bayes bad? If yes, under what aspects.
#### Q47 How would you develop a model to identify plagiarism?
#### Q48 How will you define the number of clusters in a clustering algorithm?
    Though the Clustering Algorithm is not specified, this question will mostly be asked in reference to
    K-Means clustering where “K” defines the number of clusters. The objective of clustering is to group 
    similar entities in a way that the entities within a group are similar to each other but the groups 
    are different from each other.

    For example, the following image shows three different groups.

    K-Mean Clustering Machine Learning Algorithm

    Within Sum of squares is generally used to explain the homogeneity within a cluster. If you plot WSS 
    for a range of number of clusters, you will get the plot shown below. The Graph is generally known as 
    Elbow Curve.
    
    Data Science Interview Questions K Mean Clustering

    Red circled point in above graph i.e. Number of Cluster =6 is the point after which you don’t see any 
    decrement in WSS. This point is known as bending point and taken as K in K – Means.

    This is the widely used approach but few data scientists also use Hierarchical clustering first to 
    create dendograms and identify the distinct groups from there.
#### Q49 Is it better to have too many false negatives or too many false positives?
#### Q50 Is it possible to perform logistic regression with Microsoft Excel?
#### Q51 What do you understand by Fuzzy merging ? Which language will you use to handle it?
#### Q51 What is the difference between skewed and uniform distribution?
#### G52 You created a predictive model of a quantitative outcome variable using multiple regressions. What are the steps you would follow to validate the model?
    Since the question asked, is about post model building exercise, we will assume that you have 
    already tested for null hypothesis, multi collinearity and Standard error of coefficients.
    
    Once you have built the model, you should check for following –
- Global F-test to see the significance of group of independent variables on dependent variable
- R^2
- Adjusted R^2
- RMSE, MAPE

In addition to above mentioned quantitative metrics you should also check for-
- Residual plot
- Assumptions of linear regression 
#### Q54 What do you understand by Hypothesis in the content of Machine Learning?
#### Q55 What do you understand by Recall and Precision?
#### Q56 How will you find the right K for K-means?
#### Q57 Why L1 regularizations causes parameter sparsity whereas L2 regularization does not?
    Regularizations in statistics or in the field of machine learning is used to include some extra 
    information in order to solve a problem in a better way. L1 & L2 regularizations are generally used 
    to add constraints to optimization problems.
    In the example shown above H0 is a hypothesis. If you observe, in L1 there is a high likelihood to 
    hit the corners as solutions while in L2, it doesn’t. So in L1 variables are penalized more as compared
    to L2 which results into sparsity.
    In other words, errors are squared in L2, so model sees higher error and tries to minimize that squared 
    error.
#### Q58 How can you deal with different types of seasonality in time series modelling?
#### Q59 In experimental design, is it necessary to do randomization? If yes, why?
#### Q60 What do you understand by conjugate-prior with respect to Naïve Bayes?
#### Q61 Can you cite some examples where a false positive is important than a false negative?
    Before we start, let us understand what are false positives and what are false negatives.
    False Positives are the cases where you wrongly classified a non-event as an event a.k.a Type I error.
    And, False Negatives are the cases where you wrongly classify events as non-events, a.k.a Type II error.
    In medical field, assume you have to give chemo therapy to patients. Your lab tests patients for certain 
    vital information and based on those results they decide to give radiation therapy to a patient.
    Assume a patient comes to that hospital and he is tested positive for cancer (But he doesn’t have cancer) 
    based on lab prediction. What will happen to him? (Assuming Sensitivity is 1)

    One more example might come from marketing. Let’s say an ecommerce company decided to give $1000 Gift 
    voucher to the customers whom they assume to purchase at least $5000 worth of items. They send free voucher 
    mail directly to 100 customers without any minimum purchase condition because they assume to make at 
    least 20% profit on sold items above 5K.

    Now what if they have sent it to false positive cases? 
#### Q62 Can you cite some examples where a false negative important than a false positive?
    Assume there is an airport ‘A’ which has received high security threats and based on certain 
    characteristics they identify whether a particular passenger can be a threat or not. Due to shortage 
    of staff they decided to scan passenger being predicted as risk positives by their predictive model.
    What will happen if a true threat customer is being flagged as non-threat by airport model?
    
    Another example can be judicial system. What if Jury or judge decide to make a criminal go free?
    
    What if you rejected to marry a very good person based on your predictive model and you happen to
    meet him/her after few years and realize that you had a false negative?
#### Q63 Can you cite some examples where both false positive and false negatives are equally important?
    In the banking industry giving loans is the primary source of making money but at the same time if 
    your repayment rate is not good you will not make any profit, rather you will risk huge losses.
    Banks don’t want to lose good customers and at the same point of time they don’t want to acquire 
    bad customers. In this scenario both the false positives and false negatives become very important 
    to measure.
#### Q64 Can you explain the difference between a Test Set and a Validation Set?
    Validation set can be considered as a part of the training set as it is used for parameter selection
    and to avoid Overfitting of the model being built. On the other hand, test set is used for testing 
    or evaluating the performance of a trained machine leaning model.

    In simple terms ,the differences can be summarized as-
    
    Training Set is to fit the parameters i.e. weights.
    Test Set is to assess the performance of the model i.e. evaluating the predictive power and generalization.
    Validation set is to tune the parameters.
#### Q65 What makes a dataset gold standard?
#### Q66 What do you understand by statistical power of sensitivity and how do you calculate it?
    Sensitivity is commonly used to validate the accuracy of a classifier (Logistic, SVM, RF etc.). 
    Sensitivity is nothing but “Predicted TRUE events/ Total events”. True events here are the events
    which were true and model also predicted them as true.
    Calculation of seasonality is pretty straight forward-
    Seasonality = True Positives /Positives in Actual Dependent Variable
    Where, True positives are Positive events which are correctly classified as Positives.
#### Q67 What is the importance of having a selection bias?
#### Q68 Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa.
    SVM and Random Forest are both used in classification problems.
    
    a)      If you are sure that your data is outlier free and clean then go for SVM. It is the 
    opposite - if your data might contain outliers then Random forest would be the best choice
    b)      Generally, SVM consumes more computational power than Random Forest, so if you are constrained 
    with memory go for Random Forest machine learning algorithm.
    c)  Random Forest gives you a very good idea of variable importance in your data, so if you want to 
    have variable importance then choose Random Forest machine learning algorithm.
    d)      Random Forest machine learning algorithms are preferred for multiclass problems.
    e)     SVM is preferred in multi-dimensional problem set - like text classification
    but as a good data scientist, you should experiment with both of them and test for accuracy or rather 
    you can use ensemble of many Machine Learning techniques.
#### Q69 What do you understand by feature vectors?
#### Q70 How do data management procedures like missing data handling make selection bias worse?
    Missing value treatment is one of the primary tasks which a data scientist is supposed to do
    before starting data analysis. There are multiple methods for missing value treatment. If not
    done properly, it could potentially result into selection bias. Let see few missing value treatment
    examples and their impact on selection-
    Complete Case Treatment: Complete case treatment is when you remove entire row in data even if one 
    value is missing. You could achieve a selection bias if your values are not missing at random and 
    they have some pattern. Assume you are conducting a survey and few people didn’t specify their gender.
    Would you remove all those people? Can’t it tell a different story?

    Available case analysis: Let say you are trying to calculate correlation matrix for data so you might 
    remove the missing values from variables which are needed for that particular correlation coefficient.
    In this case your values will not be fully correct as they are coming from population sets.

    Mean Substitution: In this method missing values are replaced with mean of other available values.
    This might make your distribution biased e.g., standard deviation, correlation and regression are mostly 
    dependent on the mean value of variables.

    Hence, various data management procedures might include selection bias in your data if not chosen correctly.

#### Q71 What are the advantages and disadvantages of using regularization methods like Ridge Regression?
#### Q72 What do you understand by long and wide data formats?
#### Q73 What do you understand by outliers and inliers? What would you do if you find them in your dataset?
#### Q74 Write a program in Python which takes input as the diameter of a coin and weight of the coin and produces output as the money value of the coin.
#### Q75 What are the basic assumptions to be made for linear regression?
    Normality of error distribution, statistical independence of errors, linearity and additivity.
#### Q76 Can you write the formula to calculat R-square?
    R-Square can be calculated using the below formular -
    1 - (Residual Sum of Squares/ Total Sum of Squares)
#### Q77 What is the advantage of performing dimensionality reduction before fitting an SVM?
    Support Vector Machine Learning Algorithm performs better in the reduced space. It is beneficial to 
    perform dimensionality reduction before fitting an SVM if the number of features is large when 
    compared to the number of observations.
#### Q78 How will you assess the statistical significance of an insight whether it is a real insight or just by chance?
    Statistical importance of an insight can be accessed using Hypothesis Testing.
    
    
[Reference from dezyre](https://www.dezyre.com/article/100-data-science-interview-questions-and-answers-general-for-2017/184 "悬停显示")

[Rererence from Springbord](https://www.springboard.com/blog/machine-learning-interview-questions/?from=message&isappinstalled=0 "悬停显示")
