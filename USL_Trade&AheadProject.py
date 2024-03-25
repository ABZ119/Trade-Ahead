#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning: Trade&Ahead
# 
# **Marks: 60**

# ### Context
# 
# The stock market has consistently proven to be a good place to invest in and save for the future. There are a lot of compelling reasons to invest in stocks. It can help in fighting inflation, create wealth, and also provides some tax benefits. Good steady returns on investments over a long period of time can also grow a lot more than seems possible. Also, thanks to the power of compound interest, the earlier one starts investing, the larger the corpus one can have for retirement. Overall, investing in stocks can help meet life's financial aspirations.
# 
# It is important to maintain a diversified portfolio when investing in stocks in order to maximise earnings under any market condition. Having a diversified portfolio tends to yield higher returns and face lower risk by tempering potential losses when the market is down. It is often easy to get lost in a sea of financial metrics to analyze while determining the worth of a stock, and doing the same for a multitude of stocks to identify the right picks for an individual can be a tedious task. By doing a cluster analysis, one can identify stocks that exhibit similar characteristics and ones which exhibit minimum correlation. This will help investors better analyze stocks across different market segments and help protect against risks that could make the portfolio vulnerable to losses.
# 
# 
# ### Objective
# 
# Trade&Ahead is a financial consultancy firm who provide their customers with personalized investment strategies. They have hired you as a Data Scientist and provided you with data comprising stock price and some financial indicators for a few companies listed under the New York Stock Exchange. They have assigned you the tasks of analyzing the data, grouping the stocks based on the attributes provided, and sharing insights about the characteristics of each group.
# 
# ### Data Dictionary
# 
# - Ticker Symbol: An abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market
# - Company: Name of the company
# - GICS Sector: The specific economic sector assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# - GICS Sub Industry: The specific sub-industry group assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# - Current Price: Current stock price in dollars
# - Price Change: Percentage change in the stock price in 13 weeks
# - Volatility: Standard deviation of the stock price over the past 13 weeks
# - ROE: A measure of financial performance calculated by dividing net income by shareholders' equity (shareholders' equity is equal to a company's assets minus its debt)
# - Cash Ratio: The ratio of a  company's total reserves of cash and cash equivalents to its total current liabilities
# - Net Cash Flow: The difference between a company's cash inflows and outflows (in dollars)
# - Net Income: Revenues minus expenses, interest, and taxes (in dollars)
# - Earnings Per Share: Company's net profit divided by the number of common shares it has outstanding (in dollars)
# - Estimated Shares Outstanding: Company's stock currently held by all its shareholders
# - P/E Ratio: Ratio of the company's current stock price to the earnings per share 
# - P/B Ratio: Ratio of the company's stock price per share by its book value per share (book value of a company is the net difference between that company's total assets and total liabilities)

# ## Importing necessary libraries and data

# In[4]:


#Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().system('pip install yellowbrick')

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist, pdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# to perform PCA
from sklearn.decomposition import PCA

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

#format numeric data for easier readability
pd.set_option(
    "display.float_format", lambda x: "%.2f" % x
)  # to display numbers rounded off to 2 decimal places

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# To supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[5]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=12)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="viridis",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# In[6]:


# function to plot a boxplot and a histogram along the same scale

def histogram_boxplot(data, feature, figsize=(16, 6), kde=False, bins=None, hue=None):
    """
    Combines boxplot and histogram

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (16,6))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True,
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter",
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[7]:


#reading csv file into a pandas Dataframe
ta = pd.read_csv('DATA/stock_data.csv')
# copying data to another varaible to preserve original data
df = ta.copy()


# ## Data Overview
# 
# - Observations
# - Sanity checks

# In[8]:


# print a sample of five rows randomly selected from the training data
df.sample(n=5)


# In[9]:


df.shape


# - Dataset has 340 rows and 15 columns

# In[10]:


# print the data types of the columns within the datset
df.info()


# In[11]:


# checking for duplicate values
df.duplicated().sum()


# - Dataset has no missing or duplicate values
# - All columns with dtype object should be dtype category in order to conserve memory

# In[12]:


# convert all columns with dtype object into category
for col in df.columns[df.dtypes=='object']:
    df[col] = df[col].astype('category')


# In[13]:


# dropping the ticker symbol column, as it does not provide any information
df.drop("Ticker Symbol", axis=1, inplace=True)


# In[56]:


# confirm new dataset
df.info()


# - The 14 columns have three different dtypes: category(3), float64(7), int64(4)
# - All of these dtypes are appropriate for their respective columns

# ## Exploratory Data Analysis (EDA)
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# - A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
# - A thorough analysis of the data, in addition to the questions mentioned below, should be done.

# **Questions**:
# 
# 1. What does the distribution of stock prices look like?
# 2. The stocks of which economic sector have seen the maximum price increase on average?
# 3. How are the different variables correlated with each other?
# 4. Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. How does the average cash ratio vary across economic sectors?
# 5. P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. How does the P/E ratio vary, on average, across economic sectors?

# In[15]:


#provide statistical summary of all categorical columns
df.describe(include='category').T


# In[16]:


#create labeled barplot of stocks by sector
labeled_barplot(df, 'GICS Sector')


# In[17]:


#display the five sectors with the most number of stocks
df["GICS Sector"].value_counts().head(n=5)


# - The stocks are drawn from 11 different industrial sectors, with no one sector comprising more than 16% of the dataset
# - The top 4 of the 11 sectors (industrials, financials, consumer discretionary, and health care) comprise over half of the total number of stocks
# 

# In[18]:


#create labeled barplot of stocks by sub industry
labeled_barplot(df, 'GICS Sub Industry')


# In[19]:


#display the five sub industries with the most number of stocks
df['GICS Sub Industry'].value_counts().head(n=5)


# - The dataset is comprised of stocks from 104 different subindustries, with no subindustry having more than 16 stocks in the dataset
# - These observations indicate that the 340 stocks held within the dataset are highly diversified across sectors and subindustries

# In[20]:


#provide statistical summary of all numerical columns
df.describe().T


# ### Numerical Columns

# In[21]:


#create list of columns with numerical variables
num_col = df.select_dtypes(include=np.number).columns.tolist()

#display histograms and boxplots for all numerical columns
for col in num_col:
    histogram_boxplot(df, col)


# ### Current price

# - The distribution is heavily right skewed, with 49 of the 340 stocks having twice the median value of all stocks.
# 
# - As expected, no stock is listed at less of less than 0 dollars.

# ### Price change

# - The distribution is biased towards lower volatilities, but long tails do exist both for positive and negative price changes.
# 
# - The most volatile stocks show as low as a 47% decrease to as high as a 55% increase over 13 weeks.
# 

# ### Volatility

# - As expected, the distribution of standard deviations is right skewed and not normal.

# ### Cash Ratio / ROE

# - As expected, both distributions are heavily right skewed and no stock is listed with either metric with a value of less than 0.
# 
# - For example, 24 stocks are listed with returns on equity of less than 5 and 25 stocks are listed with returns of over 100 percent.
# 

# ### Net Income / EPS

# - As expected, net income is shown to be right skewed with both long positive and negative tails
# - I.e., most companies generate meager profits, but some are failing and some are highly successful
# - 32 companies within the dataset are showing a net income of less than 0 dollars
# - EPS, as a derivative of Net Income, shows a similar distribution, with most showing low positive values and a few stocks (34) showing negative values

# ### Estimated shares outstanding

# - The distribution is highly right skewed, but no stock has a value of outstanding shares that is unrealistic

# ### P/E and P/B Ratio

# - The distribution of P/E ratios is highly right skewed
# - Interestingly, no stock shows a negative ratio, even though several stocks have a negative EPS and no stock stock has a price listed of less than 0
# - The distribution for P/B ratios is mostly centered around 0 but with long positive and negative
# - For example, 175 of the 340 total stocks are shown to below the 25th percentile and above the 75th percentile and
# - Additionally, 31 of the stocks are outliers

# ### Conclusions

# - As expected, stocks offer uncertain returns with high upsides, mostly modest returns, and the omnipresent possibility that the value of the stock may become worthless (i.e., the company goes bankrupt).
# - All of these variables contain a few or several outliers; however, none of these values appear to be unrealistic given the nature of stock prices and historical expectations.

# #### The stocks of which economic sector have seen the maximum price increase on average?

# In[22]:


df.groupby('GICS Sector')['Price Change'].mean().sort_values()


# - Stocks within the health care sectors have shown the highest average price increase over the preeceding period

# #### How are the different variables correlated with each other?

# In[23]:


#create correlation heat map for numerical variables
plt.figure(figsize=(14, 7))
sns.heatmap(
    df[num_col].corr(),
    annot=True,
    vmin=-1,
    vmax=1,
    fmt=".2f",
    cmap='viridis'
)
plt.show()


# - Several variables are moderately correlated (+/- .40) with one another
# - Volatility is negatively correlated with price change, i.e., as a stock becomes more volatile, its price is likely dropping
# - Net income is negatively correlayed with volatility, i.e. as a company generates higher net income its price is likely less volatile
# - Net income is also positively correlated with earnings per share (EPS) and estimated shares outstanding
# - EPS is positively correlated with current price, i.e. as a company's EPS rises, its prices is also highly likely to increase
# - EPS is also negatively correlated with ROE, i.e. as a company generates more equity for shareholders, an equivalent amount of net income the following periods will generate a lower return

# #### Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. How does the average cash ratio vary across economic sectors?

# In[24]:


df.groupby('GICS Sector')['Cash Ratio'].mean().sort_values(ascending=False)


# - IT and Telecommunications sectors, both relatively newer and unregulated industries, are able to generate significantly higher average cash ratios than their peer sectors
# - Utilities, a highly regulated industry, generates the lowest average cash ratios of all sectors
# 

# #### P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. How does the P/E ratio vary, on average, across economic sectors?

# In[25]:


df.groupby('GICS Sector')['P/E Ratio'].mean().sort_values(ascending=False)


# - Energy companies have the highest average P/E ratios of all sectors by a considerable margin, with telecoms having the lowest average P/E ratios

# ## K-means Clustering

# In[27]:


#scale the data set before clustering
scaler = StandardScaler()
subset = df[num_col].copy()
subset_scaled = scaler.fit_transform(subset)


# In[28]:


#create a dataframe from the scaled data
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)


# In[29]:


#create pairplot for scaled dataframe
sns.pairplot(subset_scaled_df, height=2,aspect=2 , diag_kind='kde')
plt.show()


# In[30]:


#print average distortions for range of kmeans models fitted to scaled dataset
clusters = range(1, 11)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(subset_scaled_df)
    prediction = model.predict(subset_scaled_df)
    distortion = (
        sum(
            np.min(cdist(subset_scaled_df, model.cluster_centers_, "euclidean"), axis=1)
        )
        / subset_scaled_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)


# In[31]:


#fit KMeans model and use visualizaer to indicate optimal K value
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1, 11), timings=True)
visualizer.fit(subset_scaled_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure
plt.show()


# In[32]:


#fit KMeans model and provide silhouette scores for range of k clusters
sil_score = []
cluster_list = range(2, 11)
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    preds = clusterer.fit_predict((subset_scaled_df))
    score = silhouette_score(subset_scaled_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, the silhouette score is {})".format(n_clusters, score))

#show scores in line graph
plt.plot(cluster_list, sil_score)
plt.show()


# In[33]:


#fit KMeans model and use visualizaer to indicate optimal K value
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 11), metric="silhouette", timings=True)
visualizer.fit(subset_scaled_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure
plt.show()


# In[34]:


#find optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(5, random_state=42))
visualizer.fit(subset_scaled_df)
visualizer.show()
plt.show()


# In[35]:


#create kmeans cluster model
kmeans = KMeans(n_clusters=5, random_state=42)

#fit model to scaled dataset
kmeans.fit(subset_scaled_df)


# - Between the Elbow and Silhouette plots, the number of clusters with the best performance appears to be 5

# ### Cluster Profiling

# In[36]:


# adding kmeans cluster labels to the original dataframe
df["KMeans_clusters"] = kmeans.labels_


# In[37]:


#group dataset by kmeans cluster labels
cluster_profile = df.groupby("KMeans_clusters").mean()

#add counts for number of stocks in each cluster
cluster_profile["Count"] = (
    df.groupby("KMeans_clusters")["Current Price"].count().values
)


# In[38]:


cluster_profile.style.highlight_max(color="lightblue", axis=0)


# In[39]:


# print the names of the companies in each cluster
for cl in df["KMeans_clusters"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df[df["KMeans_clusters"] == cl]["Security"].unique().to_list())
    print()


# In[40]:


#print number of stocks within each sector for all of the clusters
for k in range(0,df['KMeans_clusters'].nunique()):
    print('The number of stocks within each GICS Sector for Cluster '+str(k)+' are:')
    print(df[df['KMeans_clusters']==k]['GICS Sector'].value_counts())
    print("   ")


# In[41]:


# show boxplots of numerical variables for each K-Means cluster
fig, axes = plt.subplots(3, 4, figsize=(20, 20))
counter = 0

for ii in range(3):
    for jj in range(4):
        if counter < 11:
            sns.boxplot(
                ax=axes[ii][jj],
                data=df,
                y=df.columns[3+counter],
                x="KMeans_clusters",
                palette="viridis"
            )
            counter = counter + 1

fig.tight_layout(pad=3.0)


# ### K-Means Clusters

# ##### Cluster 0 - Large Market Capitalization / Dow Jones Industrial Average
# 
# - 11 stocks, comprised mostly of stocks within the Financials, Health Care, Information Technology (IT), and Consumer Discretionary sectors
# - Companies within this cluster have:
# - Low volatility
# - Most of the companies with the highest outflows of cash
# - The highest net incomes
# - The highest number of shares outstanding
# 
# ##### Cluster 1 - "Cash is King"
# 
# - 13 stocks, comprised mostly of stocks within the Healthcare and IT sectors
# - Companies within this cluster have:
# - Moderate volatility
# - Mostly profitable
# - Most of the highest cash ratios and cash inflows
# 
# ##### Cluster 2 - S&P 500 / Diversification
# 
# - 280 stocks (~84% of all stocks in the dataset) drawn from all sectors present in the dataset
# - Companies within this cluster have:
# - Low P/E ratios
# - Most of the outliers on negative P/B ratios
# 
# ##### Cluster 3 - "Ride the Energy Rollercoaster" portfolio / Growth mindset
# 
# - 29 stocks, a vast majority of which are from the Energy sector
# - Companies within this cluster have:
# - Low stock prices, but high ROE
# - High beta
# - Most of the most volatile stocks, especially those with outliers in price decreases
# - Mostly negative net incomes and earnings per share
# 
# ##### Cluster 4 - High Earnings for a High Price
# 
# - 7 stocks, comprised mostly of stocks from the Health Care and Consumer Discretionary sectors
# - Companies within this cluster have:
# - Most of stocks with the highest prices
# - Favorable cash ratios
# - The most favorable P/B ratios
# - Most of the highest earnings-per-share

# ## Hierarchical Clustering

# In[42]:


# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(subset_scaled_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(subset_scaled_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, round(c,4)
            )
        )
        print(" ")
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm


# In[43]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        round(high_cophenet_corr,4), high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# In[44]:


# list of linkage methods for euclidean distance metric
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(subset_scaled_df, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(subset_scaled_df))
    print(
            "Cophenetic correlation for Euclidean distance and {} linkage is {}.".format(
                lm, round(c,4)
            )
        )
    print(" ")
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm


# In[45]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        round(high_cophenet_corr,4), high_dm_lm[1]
    )
)


# In[46]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(subset_scaled_df, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(subset_scaled_df))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# - The cophenetic correlation is highest for average and centroid linkage methods, but the dendrogram for average appears to provide better clusters
# - 5 appears to be the appropriate number of clusters for the average linkage method

# In[47]:


Z = linkage(subset_scaled_df, metric='euclidean', method='average')
c, coph_dists = cophenet(Z , pdist(subset_scaled_df))


# In[48]:


hierarchy = AgglomerativeClustering(n_clusters=5, affinity='euclidean',  linkage='average')
hierarchy.fit(subset_scaled_df)


# #### Cluster Profiling

# In[49]:


df_hierarchy = df.copy()
df_hierarchy.drop("KMeans_clusters", axis=1, inplace=True)
df_hierarchy['HC_clusters'] = hierarchy.labels_


# In[50]:


#group dataset by Hierarchical clusters
cluster_profile_h = df_hierarchy.groupby("HC_clusters").mean()

#add counts for number of stocks in each cluster
cluster_profile_h["Count"] = (
    df_hierarchy.groupby("HC_clusters")["Current Price"].count().values
)

#show dataframe with maximum values for each metric highlighted
cluster_profile_h.style.highlight_max(color="lightblue", axis=0)


# - There are 2 clusters of one company, 2 clusters of two companies, and a single cluster of the remaining 334 companies
# - The clustering of these companies does not solve the business problem at hand, because the clusters do not have enough variability
# 

# #### In contrasts, the dendrogram for Ward linkage appears to provide better clustering, with 5 appearing to be the appropriate number of clusters

# In[51]:


HCmodel = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
HCmodel.fit(subset_scaled_df)


# #### Cluster Profiling

# In[52]:


subset_scaled_df["HC_clusters"] = HCmodel.labels_
df_hierarchy["HC_clusters"] = HCmodel.labels_


# In[53]:


#group dataset by Hierarchical clusters
cluster_profile_h = df_hierarchy.groupby("HC_clusters").mean()

#add counts for number of stocks in each cluster
cluster_profile_h["Count"] = (
    df_hierarchy.groupby("HC_clusters")["Current Price"].count().values
)

#show dataframe with maximum values for each metric highlighted
cluster_profile_h.style.highlight_max(color="lightblue", axis=0)


# In[54]:


# print the names of the companies in each cluster
for cl in df_hierarchy["HC_clusters"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df_hierarchy[df_hierarchy["HC_clusters"] == cl]["Security"].unique().to_list())
    print()


# In[55]:


# print the number of stocks in each GICS sector for each cluster
for k in range(0,df_hierarchy['HC_clusters'].nunique()):
    print('The number of stocks within each GICS Sector for Cluster '+str(k)+' are:')
    print(df_hierarchy[df_hierarchy['HC_clusters']==k]['GICS Sector'].value_counts())
    print("   ")


# In[ ]:


# show boxplots of numerical variables for each Hierarchical cluster
fig, axes = plt.subplots(3, 4, figsize=(20, 20))
counter = 0

for ii in range(3):
    for jj in range(4):
        if counter < 11:
            sns.boxplot(
                ax=axes[ii][jj],
                data=df_hierarchy,
                y=df_hierarchy.columns[3+counter],
                x="HC_clusters",
                palette="viridis"
            )
            counter = counter + 1

fig.tight_layout(pad=3.0)


# ### Hierarchical Clusters

# #### Cluster 0 - Growth for a Price
# 
# - 15 stocks, comprised mostly of stocks within the Health Care, Information Technology (IT), and Consumer --Discretionary sectors
# - Companies within this cluster have:
# - Most of stocks with the highest prices
# - Significant outliers in price-to-equity ratio
# - The most favorable price-to-book (P/B) ratios
# - Most of the highest cash ratios
# 
# #### Cluster 1 - Short-term Poor, Long-term Rich
# 
# - 7 stocks, comprised mostly of stocks within the Consumer Staples and Energy sectors
# - Companies within this cluster have:
# - The highest returns-on-equity
# - The lowest net incomes
# - Mostly negative earnings per share
# 
# #### Cluster 2- DJIA
# 
# - 11 stocks, comprised mostly of stocks within the Financials and Telecommunications sectors
# - Companies within this cluster have:
# - Most of the companies with the highest inflows and outflows of cash
# - The highest net incomes
# - The highest number of shares outstanding
# 
# #### Cluster 3 - Diversification
# 
# - 285 stocks (~84% of all stocks in the dataset) drawn from all sectors present in the dataset
# - Companies within this cluster have:
# - Most of outliers in price increases and some of the outliers in price decreases
# - Some of outliers in cash inflows and outflows
# - Most of the outliers in P/B ratio
# 
# #### Cluster 4 - Energy-specific portfolio
# 
# - 22 stocks, a vast majority of which are in the Energy sector
# - Companies within this cluster have:
# - Most of the most volatile stocks, especially those with outliers in price decreases
# - Mostly negative net incomes and earnings per share

# ## K-means vs Hierarchical Clustering

# You compare several things, like:
# ### Which clustering technique took less time for execution?
# 
# - Both the KMeans model and the Agglomerative Clustering model fit the dataset within ~0.1s
# 
# ### Which clustering technique gave you more distinct clusters, or are they the same?How many observations are there in the similar clusters of both algorithms?
# 
# - Both algorithms give similar clusters, with a single cluster of a majority of the stocks and the remaining four clusters containing 7-29 stocks
# 
# ### How many clusters are obtained as the appropriate number of clusters from both algorithms?
# 
# - For both algorithms, 5 clusters provided distinct clusters with sufficient observations in each to reasonably differentiate which "type" of stock is representative of the cluster
# 
# ### Differences or similarities in the cluster profiles from both the clustering techniques
# 
# - Both algorithms yielded similar clusters based on the outliers within the 11 variables
# 

# ## Actionable Insights and Recommendations
# 
# - 

# Trade&Ahead should initially assess their clients' financial objectives, risk tolerance, and investment behaviors. Based on this information, they can suggest a cluster of stocks that aligns with these requirements, potentially serving as a suitable portfolio.
# 
# However, it is important to note that many of these clusters, characterized by specific stock attributes, essentially serve as alternatives to widely recognized indexes like the Dow Jones Industrial Average and the S&P 500. These indexes may offer a more straightforward means of achieving the desired objectives.
# 
# Alternatively, Trade&Ahead could utilize these clusters as a starting point for conducting in-depth financial statement analysis, particularly focusing on individual stocks that do not conform to the cluster's "profile." If the selection of individual stocks is part of a client's investment strategy, Trade&Ahead may be able to identify stocks that have the potential to outperform their peers (i.e., with a buy recommendation, anticipating price appreciation) or stocks that are likely to underperform their peers (i.e., with a sell recommendation, anticipating price depreciation).
