# Sparkify
> TL;DR: The goal was to use Spark for the first time and predict the churn rate of users on the data set of Sparkify. The result turned out better then expected.

Spark is one of those names that often comes up when you are engaged in machinelle learning. I used the opportunity of my Udacity nanodegree to familiarize myself with this new technology.

The goal of this project is to predict if a user will downgrade or leave the service *Sparkify*. *Sparkify* is a music company which lets you play music, comparable to Spotify. They provide two different levels for users. In the free mode, publicity is played between different songs. If you pay for the service, no advertisement is played. 

A Sparkify data set was provided for the project. For my implementation I only used the provided mini subset, because my goal was primarily about trying out the technology Spark. Good results in machine learning algorithms were only secondary. All the better that good prediction results came out of it.


## How does the data look like?
Sparkify is a music provider which wants to predict the churn rate of a user. 

The information in the data can be separated into 3 parts: general, user and current song. 
| General        | User          | Current Song |
|----------------|---------------|--------------|
| auth, itemInSession, method, page, status | firstName, lastName, gender, level, location, registration, session_id, ts, userAgent, userId | artist, length, song     |
||||


Below you can see a example row of the data set.

```
Row(artist='Martha Tilston', auth='Logged In', firstName='Colin', gender='M', itemInSession=50, lastName='Freeman', length=277.89016, level='paid', location='Bakersfield, CA', method='PUT', page='NextSong', registration=1538173362000, sessionId=29, song='Rockpools', status=200, ts=1538352117000, userAgent='Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0', userId='30'),
```
For further information on the different data fields, see the collapsed part below.

<details><summary>Details for all data columns </summary>
<p>

| Column        | Description |
|---------------|-------------|
| artist        | Performer of the song|
| auth          | Is the user logged in or not?|
| firstName     | First name of the user|
| gender        | Gender of the user|
| itemInSession | How many songs has the user already listened to in this session?|
| lastName      | Family name of the user|
| length        | Length of the currently played song|
| level         | Does the user use a free or paid account?|
| location      | Where is the user located? |
| method        | HTTP Method - PUT or GET, depending on the page|
| page          | Which page is opened by the user? - Home, Settings, NextSong, ...|
| registration  | Timestamp when the user has registered|
| sessionId     | Userspecific id of the session|
| song          | titel of the currently played song|
| status        | HTTP status - 200 or 404|
| ts            | timestamp when this action was performed|
| userAgent     | which browser does the |
| userId        | unique id of the user|
</p>
</details>

## How is the data preprocessed?
As mentioned in the introduction, the goal of this implementation is to build a machine learning algorithm which can predicted if a user is going to downgrade or leave the service.

Some of the information provided in the data set are not important or relevant for this kind of prediction. 
All informations related to the current song (*artist*, *length*, *song*) are irrelevant, because the music taste of a user should not influence his churn behaviour.

The user-related columns *firstName*, *lastName* and *gender* could misguide the machine learning algorithms, therefore they will not be considered further.

### Adaptation of the user_id
As it can be seen in the following figure, a user can have several downgrade events during his registration period. But the system can only predict one downgrade event. So the user journey was split into different chunks. Each chunk has a own id. To keep a human-readable relation between the chunks and the corresponding user, a suffix was added for the user_id.

![alt text](overview_user_id.jpg "Logo Title Text 1")

### Splitting the data into two data sets
The data can be viewed in two different ways. On the one hand, the data can be viewed based on the different interactions of a user, i.e. his customer journey, which leads to a downgrade. This data is viewed in the *session_df* data frame.

On the other hand, central facts of a user can also be viewed independently of his individual sessions. These are collected in the data frame *user_df*.
#### Preprocessing of *session_df*
For the time based processing, the following columns might be interesting: *userId*, *sessionId*, *level*, *itemInSession*, *location*, *status*, *ts*, *currSessionLength*.
*userId*, *sessionId*, *level*, *itemInSession* and *ts* were not further preprocessed, since they could already be used in their present form.
In the following table, it is explained, what kind of preprocessing has been performed for the remaining columns.
| Column        | Idea/Assumption | Preprocessing |
|---------------|------|---------------|
| location      | Does the user change its location during a session, would this be a sign that he travels. As no user has several locations in one session, no one travels. So the column is removed from the data frame as it does not contain valuable information.|*column was dropped*|
| status        |If a user has often problems with a website, the probability that he will downgrade is higher.| Converted into a binary error column.|
| currSessionLength | The longer the duration of a session, the more a user interacts with the service, this means that he is probably satisfied with the service and will not downgrade.| Calculate the difference between the last and the first timestamp of a session. |

#### Preprocessing of *user_df*
For a user-centered prediction, the following columns of the original data are of interest: *userId*, *sessionId*, *level*, *ts*, *page*, *registration* and *userAgent*.
These columns are used to determine the following values for each user: *level*, *nb_devices*, *avgSessionLength*, *lastInteraction* and *freqOfUse*.

| Column           | Idea/Assumption | Preprocessing |
|------------------|------|--------------|
| nb_devices       | The more different devices uses, the more he likes the service. As every user only uses one device, the column is deleted.|*column was dropped*|
| avgSessionLength | The longer a user's sessions are, the less likely it is that the user will downgrade the service.| Calculate the length of each session of this user, then take the mean.|
| lastInteraction  | If a user has not used the service for a long time, he will probably downgrade. | Biggest timestamp of this user|
| freqOfUse        | At how many days did the user use the service?|Convert the timestamps into a duration and calculate the number of days. Then set this value in relation to the number of days in the entire data record. |


## How can the churn rate be predicted?

### Modelling with *session_df*
Here is a climpse of the preprocessed *session_df* data frame:
```
   userId  sessionId  level  itemInSession             ts  downgrade  
0   30_0         29      1             50  1538352117000          0   
1    9_0          8      0             79  1538352180000          0   
2   30_0         29      1             51  1538352394000          0   
3    9_0          8      0             80  1538352416000          0   
4   30_0         29      1             52  1538352676000          0   
```
Unfortunately, PySpark does not provide any machine learning algorithm for time series based prediction, so this part has not been continued. I added the explaination of this data preparation to document another possibility what could be done with another machine learning technology.

### Modelling with *user_df*
Here is a small excerpt of the preprocessed data, which is now used for the machine learning algorithms.
```
   userId  level  avgSessionLength  lastInteraction  freqOfUse  downgrade
0   30_0      1      1.889567e+07    1538995454000   0.063444          1
1    9_0      0      6.605500e+06    1538839066000   0.063444          0
2   74_0      0      8.611714e+06    1539939591000   0.095165          0
3   54_0      1      2.944917e+07    1539608060000   0.174470          1
4    4_0      0      2.950778e+06    1540121320000   0.142748          0
```
The task of this project was to compare the performance of three different machine learning algorithms. The following have been choosen:
- Logistic Regression
- Random Forest
- Gradient-boosted Tree Classifier

To make the three different ML Algorithms as comparable as possible, the random split of training and validation data and the reformatting of the features and labels has been defined one time before the implementation of the algorithms. All Parameters for the mentioned parts are directly defined, they are not modified in any of the ParamGrids.

For the Evaluation of the models, the f1-score is used.

All three models have been trained using a pipeline and a crossvalidator to reduce the need of adaption. In this table it can be seen which values have been choosen for optimizing:
|Logistic Regression | Random Forest | Gradient-boosted Tree |
|--|--|--|
|maxIter :  [10, 100, 200]        |maxDepth: [2, 5, 10, 30]|maxDepth: [2, 5, 10, 30]|
|regParam:  [0.0, 0.1, 0.5]       |numTrees: [2, 5, 10, 50]|stepSize: [0.03, 0.1, 0.5, 0.9]|
|threshold: [0.1, 0.5, 0.95, 1.0] |maxBins:  [2, 5, 10, 50]|maxBins:  [2, 5, 10]|
My goal was to choose the parameter which have the highest impact on the model.

In the following table, the result of the best models are shown. The values for the F1-Score are rounded for 4 decimal places.

| Algorithm | Parameters | F1-Train | F1-Test |
|---|---|---|---|
| Logistic Regression | maxIter: 10, regParam: 0.0, threshold: 0.5 | 0.7657 | 0.6471 |
| Random Forest | maxDepth: 5, numTrees: 10, maxBins: 10 | 0.7492 | 0.6748 |
| Gradient-boosted tree classifier | maxDepth: 2, stepSize: 0.03, maxBins: 5 | 0.8900  | 0.9210 |

As the table shows, an unambiguous result has been achieved. To determine if a user wants to downgrade or leave the service a Gradient-boosted tree classifier should be used.