# **Currents in Islamophobic Rhetoric**
### *Analyzing Trends in Online Anti-Muslim Discourse*
- **Author:** Mohammad Abou-Ghazala
- **Date:** January, 2026

![image](https://github.com/momopajamas/MPAC_project/blob/main/images/islamophobia_image_675.jpg?raw=true)

([*Image Source*](https://nysba.org/islamophobia-surges-in-the-u-s-due-to-global-and-national-tensions/?srsltid=AfmBOooMQkb1x343wsT1sVo5tCbEgvzjFDnUVsFsK4r-afxU8pNhnb2j))

# I. Introduction
With the upsurge in Islamophobia rhetoric in recent months that is starkly reminiscient of a post-9/11 climate in the US and the West, the Muslim Public Affairs Committee (MPAC) has tasked us with locating trends within this discourse.

## Problem Understanding
Our goal is to identify dominant themes underpinning online Islamophobic discourse to better inform research and policy recommendations. Specifically, we **focus on extracting actionable insights by identifying recurring language patterns, narratives frames, and potential changes in discourse that can support effective advocacy, monitoring, and community responses**. 

To accomplish this, we analyze a dataset of public posts from X (formerly Twitter) from a variety of users through the following:
1. Build a baseline Natural Language Processing (NLP) classifier that can identify Islamophobic tweets.
2. Use the Islamophobic tweets we identified to extract themes through Topic Modeling and measure their salience by weighing rhetoric against engagement metrics.

#### Defining Islamophobia
For the purposes of identifying anti-Muslim and anti-Islam rhetoric, we used the definition of Islamophobia used by [Georgetown University's Bridge Initiative](https://bridge.georgetown.edu/about-us/what-is-islamophobia/) which defines it as "*Islamophobia is an extreme fear of and hostility toward Islam and Muslims which often leads to hate speech, hate crimes, as well as social and political discrimination.*"

Using this more limited definition allows us to narrow the scope of our investigation.

### Project Scope
- Our dataset consists of 1,619 tweets collected from X.
- The project creates a small, high confidence manually labeled subset to train and evaluate a baseline classifier.
- The raw dataset is unlabeled, so we create a small set of 200 tweets, half labeled 0 for not Islamophobic, and the other half labeled 1 for Islamophobic.
- The modeling approach prioritizes interpretability and speed, using traditional NLP features (e.g., TF-IDF) and linear models rather than computationally expensive deep learning approaches.
- After classification, the subset of Islamophobic tweets is used to perform topic modeling, producing interpretable clusters/topics supported by representative examples and key terms.
- Due to the sensitive nature of the domain and the risk of harm from misclassification, the classifier is framed as a decision-support tool rather than an automatic enforcement system.

### Success Criteria
We will measure success for this project by posing three questions:
**1. Is the classifier functional and reliable enough to support triage?**
**2. Did we learn something real, clear, and repeatable about the discourse?**
**3. Can the insights inform real decisions?**

# II. Data Understanding
This dataset contains 1,619 public posts from X (formerly Twitter). Each row represents a single post and includes the post text, metadata about the author account, and engagement metrics.

### Features
The following columns are of most interest to us:
- `X Accounts`: The username or account identifier that published the post. We can use this to assess repition, concentatrion of posting behavior, or identify prevelance of high-volume account.
- `Post Link`: Direct URL to the post for auditing and transparency.
- `Post Text`: The main content of the tweet. **This is the primary input feature for NLP tasks such as classification and Topic Modeling.**
- `Retweets` / `Comments` / `Likes`: Engagement metrics that approximate reach and amplification. Views represent the broadest measure of exposure, while retweets may indicate stronger forms of engagement. **We can use this to help Impact Analysis by measuring salience of rhetoric**.

This dataset supports three core types of analysis:

**1. Classification**: Using the post text to train a classifier that can triage harmful content.

**2. Topic Modeling**: Identifying recurring narratives and language patterns within Islamophobic tweets (e.g., stereotypes, collective blame, exclusionary rhetoric).

**3. Impact Analysis**: Using engagement metrics, especially views and retweets, to understand which narratives are most amplified and hold higher degrees of salience, and therefore may pose greater public influence or harm.

### Quality Considerations & Limitations
Several quality issues are present in this dataset, namely:
- Context loss: Tweets are short and often depend on context (threads, quotes, sarcasm, or external links or links to other tweets). This creates ambiguity when interpreting intent.
- Noise in text: Tweet text may include URLs, mentions (@user), hashtags, emojis, or formatting artifacts that require preprocessing.
- Engagement bias: Views and engagement are influenced by account size, virality dynamics, and platform algorithms, not only the content itself, so they should be interpreted as approximate impact signals rather than direct measures of harm.
- Sampling bias: The dataset may not represent the full spectrum of Twitter discourse. It may be shaped by how the tweets were collected (search terms, accounts, timeframe).

Additionally, the scope of this project and the nature of this dataset combined to create a number of limitations:
- Our focus on identifying Islamophobia in particular leads us to exclude rhetoric containing other forms of hate and extremism, such as anti-Hindu or antisemitic rhetoric, which can often emanate from the same source or utilize overlapping forms of rhetoric.
- Some tweets are antagonostic towards ethnicities or countries, and do not constitute Islamophobia in the strict sense we outlined above.
- The vast majority of the tweets deal with the the US context specifically, which could limit the transnational dimensions of Islamophobia.

### Model Selection
For efficiency, we will deploy two models and assess how they perform on training and test data Both models will incorporate **Term Frequency-Inverse Document Frequency (TF-IDF)**, which will be useful in determining the relative significance of terms used across the content.

1. `Logistic Regression`: This will be our baseline model as it is strong in text classification and relatively quick to train. The probabilities given by this model will potentially be useful for threshold tuning, and can provide us with the top words driving predictions.
2. `LinearSVM`: This model excels at high-dimensional sparse text, and is likewise quick to train and robust, and also has a function to measure the relative confidence in each classification.

#### Cross-Validation
Since we are working with a relatiely limited set of labeled data of only 180 entries, we will need to ensure that our models are functioning competently and not by mere luck. We can do this by relying on cross-validation, which will run several data splits instead of just one, and will use `StratifiedKFold()` to conduct those splits.

#### Topic Modeling & Clustering Analysis
To extract themes from classified Islamophobic tweets, we will feed the vectors created by TF-IDF() into a Truncated Singular Value Decomposition (`TruncatedSVD()`) model, due to its ability to process sparse data efficiently and capture patterns in that data. 

Then we can use `KMeans()` to create clusters out of those modeled vectors, allowing us to see which terms cluster together and learn which themes can be gleaned from the tweets.

### Metrics
Because we are looking to distill Islamophobic tweets into repeated themes that we can extract, it would be worse for us to accidentally classify a non-Islamophobic tweet as Islamophobic, as it would contaminate our dataset and the themes we extract.

In this case, **False Positives are worse than False Negatives**.

Therefore, we will be using **Precision** as our primary metric of success, as this evaluates our ability to **keep False Positives out** and generate a high-confidence subset for analysis.

## Data Preparation
After manually labeling a small subset of 200 tweets, there are a number of steps we need to take before to clean the data:
1. Remove unnecessary columns
2. Remove data entries with null `Post Text` values, or only contain hyperlinks or photos
3. Replace hyperlinks in tweets with standard tokens
4. Address spelling errors and standardizing the text

### Preprocessing the Text
To clean up our text, we created a custom function, `clean_tweet()` for text preprocessing to perform the following steps:
1. Lowercase text
2. Replace URLs with "URL"
3. Replace @mentions with "USER"
4. Normalize quotes
5. Remove extra whitespace

### Extracting Labeled Data
After preprocessing the entrie dataset, we extract the smaller subset of labeled data and create a new dataframe in order to train our classifiers, `df_labeled`. 
- 0 = Not Islamophobic
- 1 = Islamophobic

### Splitting the Data
Our final step in preparing our data is to split `df_labeled` into training and testing sets to use with our classifiers using `train_test_split()`.

# III. Data Modeling
## Pipelines
For efficiency, we create a pipeline for each of our classifier models, incorporating `TF-IDF()` into each pipeline to vectorize the text.

We rely on **Precision** to evaluate the performance of each model and decide which to apply on the larger unlabeled dataset. 

### Cross Validation
To deepen our confidence in our models, we use cross validation via `StratifiedKFold()`, which produced the following Precision Scores:
- `Logistic Regression`: **82%**
- `LinearSVM`: **86%**

Since our `LinearSVM()` model produced a more 'pure' bucket of Islamophobic tweets, it is the model we apply on the larger dataset.

## Classifying the Entire Dataset
Using `LinearSVM()` we classify the entire dataset, adding a column titled `pred_label` to indicate classification.

Using this model's `decision_function` we create another column titled `conf_score` as an indication of the relative confidence behind the model's classification.

Then we pull the Islamophobic tweets we more more confident about into a new dataframe, `df_anti_islam`, which contains a **total of 458 tweets** to work with.

## Topic Modeling
Using `TruncatedSVD()` along with `KMeans()`, we're able to both cluster our data together and extract keywords and themes from the clusters.
1. Vectorized the data within `df_anti_islam` using TF-IDF
2. Applied `TruncatedSVD()` on the vectorized data to reduce dimensionality of the vectorized data matrix and capture linguistic patterns
3. Used `KMeans()` on the truncated data to group posts into 5 thematic clusters

## Evaluation of Themes
After reading through the top terms found in each cluster, we were able to identify 5 themes:

- **Theme 0**: a more vague, general fear of Sharia Law in the US
- **Theme 1**: heightened interest and attention on Texas, where Islamophobic personalities have stirred tensions around the growing Muslim population and a perceived increase in mosques, Specifically an agitation against the [East Plano Islamic Center (EPIC) City](https://en.wikipedia.org/wiki/EPIC_City,_Texas) project, which is a master-planned Islamic community-centered residential project in Texas
- **Theme 2**: similar to Theme 1, this cluster appears to be concerned with a particular locale, being Dearborn, MI, known for its high concentration of Muslim and Arab residents, and potentially drawing connections with the situation in Texas.
- **Theme 3**: an increased interest in the Muslim Brotherhood as well as fear of its alleged presence and activity in the US. The presence of [CAIR](https://www.cair.com) (which the the Council on American Islamic Relations) in this cluster may confirm this conclusion considering recent moves by the Governors of Florida and Texas to impose restrictions on CAIR activity under the allegation of its connection to the Muslim Brotherhood
- **Theme 4**: a more general concern with perceived Islamization of the US. The presence of "Texas" yet again may further confirm heightened attention around Texas specifically as what Islamophobes posit as a battleground of site of struggle against the alleged Islamization of America

We measure the salience of each of these themes by graphing their relative engagement according to `Likes` and `Retweets`:

![Bar Chart](https://github.com/momopajamas/MPAC_project/blob/main/images/theme_engagement_graph.png?raw=true)

Based on these findings, we synthesize our assessments into the following insights:

**1. The current state of Islamophobic discourse online is fueled to a significant degree by localized developments and concerns**, specifically Dearborn, MI and to a larger extent, Texas.

**2. Geopolitical developments in the Middle East are informing domestic attitudes towards Islam in the US**, as reflected in the prominence of tweets pertaining to perceived threats from the Muslim Brotherhood and an alleged connection with Muslim civil rights organizations.

**3. There remains, or there is a resurgence of, a more general fear of Islamization and Shariah Law**, which could feasibly be in part informed by the wider rise in xenophobia and anti-immigrant sentiments.

# Conclusion
We successfully built a binary classifier with an impressive Recall score capable of extracting Islamophobic tweets from our dataset while minimizing False Positives, and then performed Topic Modeling on those classified tweets to extract themes. 

### Insight Learned
We were able to synthesize our findings into the following insights:
1. The prominence of tweets pertaining to Dearborn and Texas tells us that **much of Islamophobic rhetoric is centered around ongoing, kinetic domestic situations and rooted in localized concerns.**
2. Presence of rhetoric regarding the Muslim Brotherhood and alleged connections to domestic groups indicates that **Islamophobic fears are at least in part informed by international developments related other the Middle East.**
3. Recurring fears of Islamization and Shariah Law points to lingering, or reactivated, **generalized fear of consequences of a growing Muslim population and its impact on the US.**

### Evaluating our Success Criteria
##### 1. Is the classifier functional and reliable enough to support triage?
Answer: ***We cannot confidently assert that our classifier is sufficiently equipped for general application and triage**. This is mainly due to limitations pertaining to the dataset itself, which will be elaborated upon in the Limitations section below.*

##### 2. Did we learn something real, clear, and repeatable about the discourse?
Answer: ***Yes**, we were able to ascertain tangible themes contained within the dataset that reflect prominent currents in Islamophobic rhetoric posted and amplified on social media.*

##### 3. Can the insights inform real decisions?
Answer: ***Yes**, we were able to understand social and political causes animating major currents in Islamophobic discourse, and can effectively use this deeper understanding to inform recommendations to interested actors.*

## Limitations
There were three categories of limitations we faced:
#### Limitation 1: Dataset
Since we were not in control of how the data was collected and are not privy to the methodology behind it, there are a number of **limitations on our investigation arising from the dataset**:
1. We do not know the degree of its veracity or how representative it is. 
2. There is an imbalance in the dates the tweets were collected, and we can't credibly investigate trends over time.
3. There is an outsized presence of a handful of virulent Islamophobic personalities, such as Amy Menk or Laura Loomer, which more than likely skewed our classifier's ability to understand the nuances of Islamophobic discourse. 
4. Many of the tweets contained photos with text that was not transcribed, so we were unable to factor in the language used on those photos in our modeling.

#### Limitation 2: Scope of the Project
Furthermore, our decision to zero in on a strict definition of Islamophobia allowed us to be more focused in our modeling, but most likely caused us to miss more nuanced topics:
1. To what extent is a xenophobic fear of certain immigrant communities tied to Islamophobia?
2. In what ways does more subtle forms of Islamophobia inform opinions on international matters?

#### Limitation 3: Time and Resources
Due to **limited hardware and time constraints we chose to prioritize efficiency** over the course of this project, impacting us in the following ways:
1. Classifier Performance - We only tested two binary classifiers over the course of this project. With more time, we could have tested a few more to compare performance. Furthermore, we could not spend time tuning the models' hyperparameters to maximize efficacy.
3. Target Labeling - We were forced to rely on a small subset of manually labeled tweets, impacting the efficacy of our models. Moreover, with a larger labeled set we could have focused more on Topic Modeling.
4. Hardware - With stronger hardware, we could have deployed more effective models on our data, such as BERT models that are specifically designed to handle social media text and tweets in particular. 

## Recommendations
Based on our synthesized insights above we confidently recommend the following:

#### Recommendation 1
Urge local and national community leaders to invest in efforts to market densely-populated Muslim communities towards non-Muslims to assuage fears and build tangible connections that can over time foster understanding. This is especially applicable to Texas in the case of the EPIC City project.

#### Recommendation 2
Intensify efforts by Muslim civil rights organizations to emphasize their 'American-ness' by investing in a communications strategy that more explicitly references American values and pillars, and by more intentionally posturing as being inseparable from of American society and body politic, and more visibly disavowing any connections to foreign entities.

#### Recommendation 3
Related to Recommendation 2, is to promote engagement in intercommunal and interfaith initiatives to foster understanding between Muslims and non-Muslims with an aim to minimize fears of Islamization and concepts like Shariah Law.

## Next Steps
We can take this investigation further in the future through the following steps:
1. Invest in hardware capable of running more sophisticated models like BERT for more thorough and efficient processing of tweets and social media posts.
2. Revamp our dataset so it is more comprehensively representative of wider discourse in the following ways.
3. Deepen the potency of our insights by locating potential intersections between Islamophobic discourse and other manifestations of hate, such as antisemitism or xenophobia, in order to apprehend any reinforcing relationships between different forms of online hate.

# Appendix
### Sources
- [Bridge Initiative's Definition of Islamophobia](https://bridge.georgetown.edu/about-us/what-is-islamophobia/)
- [El Plano Islamic Community (EPIC) City](https://en.wikipedia.org/wiki/EPIC_City,_Texas)
- [Counccil on American-Islamic Relations (CAIR)](https://www.cair.com)
## Navigation
- [Jupyter Notebook](https://github.com/momopajamas/MPAC_project/blob/main/notebook.ipynb)
- [Images & Visualizations](https://github.com/momopajamas/MPAC_project/tree/main/images)
- [Non-Technical Presentation](https://github.com/momopajamas/MPAC_project/blob/main/presentation/anti-islam_currents_presentation.pdf)
