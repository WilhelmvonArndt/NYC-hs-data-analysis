# NYC Elementary School Data Analysis

The following project was created as a final submission to Pascal Wallisch class *Intro to Data Science*.


In this project our primary focus revolves around examining whether specific characteristics associated with middle schools in NYC can serve as predictors for admission to one of the eight highly selective public high schools in the city, collectively known as HSPHS. It is important to highlight that admission to these esteemed schools necessitates not only the submission of an application but also achieving a high score on the Specialized High Schools Admissions Test (SHSAT), which is an externally developed and anonymously evaluated standardized test. Through this project, we aim to uncover valuable insights and explore the factors that may influence the admission process to these exceptional educational institutions.

#### Managing missing data

Due to imperfect data, I'll employ methods to address missing data in this project. NaN removal will be applied to necessary rows (schools) without removing all rows with >=1 nan, as our aim is to retain as much data as possible. Throughout the project, you'll observe the concatenation of smaller dataframes from the original data and the implementation of the nan-removal method on these subsets. For instance, when correlating column A with B in a dataset containing NaNs in columns A, B, and C, we'll group A and B before performing the action solely on those columns. This approach preserves rows where column C contains NaNs, rather than excluding them based on perfect data. Additionally, I utilize the nanRemover function for Numpy arrays, which is defined under section I of the appendix.


#### Determine which is a better predictor of admission to HSPHS: the raw number of applications or the application rate.

To ensure accurate analysis, schools with zero applicants were excluded from the calculation. This step was taken to avoid ambiguity, as schools with zero acceptances but non-zero applications may erroneously be considered similar to those with both zero applications and acceptances. Subsequently, two linear regression models were employed to predict admissions (Y) using the number of raw applications (x) and the application ratio (x) as predictors. The results revealed that raw applications proved to be a superior predictor, with a coefficient of determination (R^2) of 0.655 (represented by the blue line), compared to a coefficient of 0.440 for the application-rate model (represented by the red line). This suggests that the number of raw applications provides more meaningful insights in predicting admissions outcomes. Code can be found under II in the apendix.

![](graphs/1.png) ![](graphs/2.png)

#### Analyzing which school has the best per student odds of sending someone to HSPHS.

The ratio was calculated by examining the acceptances/school size ratio. The school with the highest per student odds HSPHS was Christa McAuliffe School with odds of 3.26 to 1 for sending someone to HSPHS. Code can be found under seciton III in the apendix.


#### Investigating the relationship between students' perception of their school (as reported in columns L-Q) and the school's performance on objective measures of achievement (as noted in columns V-X).

The first step to determining if there are any relationships worth investigating by plotting a correclation table:

![](graphs/3.png) 

Upon further analysis, it becomes evident that multiple correlations exist within the dataset, highlighting the need for dimension reduction. To address this, a two-dimensional Principal Component Analysis (PCA) will be conducted. Prior to conducting PCA, the data will be normalized to ensure consistency and comparability across variables.

![](graphs/4.png) ![](graphs/5.png)

To evaluate the Eigenvalues, the Kaiser criterion will be applied (1), represented by the red line. This criterion helps determine the significance of the factors. For both PCA analyses, one crucial factor is retained based on this criterion. Next, the scores of the two factor loadings will be examined:

![](graphs/6.png) ![](graphs/7.png)

When examining the loadings of Perception (shown in the left chart above), we observe four notable loadings that are of interest and warrant further analysis. These loadings correspond to Rigorous Instructions (1), Collaborative Teachers (2), Effective School Leadership (4), and Trust (6). Additionally, when investigating the loadings of student achievement (shown in the right chart), we identify one significant loading related to Reading Scores Exceeded (2). To analyze the variance among these loadings, an ANOVA is conducted, yielding the following table:

![](graphs/8.png)

The results of our model regarding the interaction effects were not particularly informative, as they lacked significance, displayed low F-values, and had limited sums of squares. However, the main effects provide more interesting findings. Notably, trust, reading scores exceeding expectations, and rigorous instruction all exhibit significant p-values, accompanied by good F-values and sums of squares (although it should be noted that the loading for rigorous instruction was relatively low). Since our initial model did not yield any significant interaction effects, a decision was made to narrow down the analysis. Consequently, a follow-up ANOVA was conducted, excluding 'rigorous instruction' and 'effective school leadership' loadings, focusing instead on loadings with an absolute value of approximately 0.4. The resulting ANOVA table is as follows:

![](graphs/9.png)

Upon analyzing the results, it becomes apparent that the significance of 'reading scores exceed' is greater than initially anticipated. Furthermore, the importance of 'trust' is confirmed as a significant factor. Additionally, an intriguing interaction effect emerges between 'collaborative teachers' and 'trust'. Based on these findings, it can be concluded that there is indeed a relationship between student perception and the measured variables. Although the strength of this relationship may not be as pronounced as the importance of reading scores, it is evident that perception does play a significant role. Code for this can be found in section IV of the appendix.



#### Testing the hypothesis whether if size of the school impacts the student's achievement.  

In this analysis, the school size is categorized into two groups, namely large and small, using the median as the dividing point. Subsequently, a scatter plot is generated to gain intuitive insights from the visual representation of the data.

![](graphs/10.png)

Upon visual inspection, there appears to be no noticeable difference in achievement between small schools (represented in red) and large schools (represented in blue). However, to delve deeper into this observation, further investigation is warranted. The null hypothesis formulated is that school size does not impact student achievement. Before selecting an appropriate test to evaluate this null hypothesis, the normal distribution of the achievement variable is assessed.

![](graphs/11.png) ![](graphs/12.png)

The achievement variable appears to follow a normal distribution. Consequently, an independent t-test is conducted to compare the achievement levels of small schools and big schools. The resulting p-value is approximately 0.06517. Since this p-value exceeds the significance threshold, we are unable to reject the null hypothesis. Therefore, based on the statistical analysis, it can be concluded that the size of the school does not have a significant impact on student achievement.

#### Developing a model that includes all factors to identify the most important school characteristics for a) sending students to HSPHS, and b) achieving high scores on objective measures of achievement.



Summarize the findings to identify the school characteristics that appear most relevant in determining acceptance of their students to HSPHS.
Provide actionable recommendations to the New York City Department of Education on how to improve schools in terms of a) increasing the number of students sent to HSPHS, and b) enhancing objective measures of achievement.

#### I
```python
def nanRemover (index1, index2, dataset):
    counter=0
    outputData =[]
    for ii in dataset[index1]:
        if np.isnan(ii) == True:
            counter+=1
            pass
        elif np.isnan(dataset[index2][counter]) == True:
            counter+=1
            pass
        else:
            outputData.append([dataset[index1][counter],dataset[index2][counter]])
            counter+=1
    outputData = np.array(outputData)
    return outputData
```

#### II

```python
apps2 = data['applications'].replace(0,np.nan)
size2 = data['school_size']
accept2 = data['acceptances']
names2 = data['school_name']

df_q2_comb = pd.concat([names2,apps2,size2,accept2],axis=1)
df_q2_comb = df_q2_comb.dropna()

appRate = df_q2_comb['applications']/df_q2_comb['school_size'] #how many students at a particular school applies.
appRate = appRate.to_frame() #predictor #1

y_admissions = df_q2_comb['acceptances'] #this will be our y a.k.a our outcome (what we're predicting)
y_admissions = y_admissions.to_frame() #Converting from series to dataframe for skl

x_rawApp = df_q2_comb['applications']
x_rawApp = x_rawApp.to_frame() #Dataframe with our raw applications, predictor #2

regr_appRaw = linear_model.LinearRegression()
regr_appRaw.fit(x_rawApp,y_admissions)

appRaw_r_sq = regr_appRaw.score(x_rawApp,y_admissions) #getting the r^2 value of raw applications
#plotting Raw applications and # of admissions


plt.plot(x_rawApp, regr_appRaw.predict(x_rawApp), color='blue', linewidth=1)
plt.title('Raw admission and accepted students, blue line = linear regression ')
plt.ylabel('Accepted students')
plt.xlabel('Number of Admissions')
plt.show()


tempComb = pd.concat([y_admissions,appRate],axis=1)
tempComb = tempComb.to_numpy()
tempComb = tempComb.transpose()
tempComb = nanRemover(0,1,tempComb) #removes nan and outputs a 2d array that will be usable for our linear regression
tempComb = tempComb.transpose() #transpose for easier indexing, [0]=y; [1]=x

appRate_nonan = tempComb[1].reshape(-1,1) #reshaping for fitting the linear regression model
y_admissions_nonan = tempComb[0].reshape(-1,1)

regr_appRatio = linear_model.LinearRegression()
regr_appRatio.fit(appRate,y_admissions)

plt.scatter(appRate_nonan,y_admissions_nonan,color='black',s=7)
plt.plot(appRate_nonan, regr_appRatio.predict(appRate_nonan), color='red', linewidth=1)
plt.ylabel('Accepted students')
plt.xlabel('Ratio of students applying/students accepted')
plt.title('Ratio and accepted students, red line = linear regression ')
plt.show()
    
modelRatio = sm.OLS(y_admissions,appRate).fit()
predictionsRatio = modelRatio.predict(appRate)

print(modelRatio.summary())

modelRaw = sm.OLS(y_admissions,x_rawApp).fit()
predictionsRaw = modelRaw.predict(x_rawApp)


```
#### III
```python
df_admission_rate = (data['acceptances']/data['school_size']).to_frame() #acceptances over school size gives us ratio
df_admission_rate.columns = ['admission_rate']
df_school_names = (data['school_name']).to_frame()
df_admissionrate_schools = (pd.concat([df_school_names,df_admission_rate], join='outer',axis=1)).dropna()

x=1/0.235
y=x-1
```

#### IV
```python

df_perception = data[data.columns[11:17]] #columns L-Q
df_achievement = data[data.columns[21:24]] #columns V-Z

df_contemp = (pd.concat([df_perception,df_achievement], join='outer',axis=1)).dropna() #removes nan necessary

df_perception_nonan = df_contemp[df_contemp.columns[:6]] #slices
df_achievement_nonan = df_contemp[df_contemp.columns[6:]]

df_perception_performance = df_contemp.copy()

#Creating a correlation matrix

r4 = np.corrcoef(df_perception_performance,rowvar=False)
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 
plt.title('Correlation table Student Perception and Achievement')
plt.imshow(r4)
plt.colorbar()

#DOING PCA for Students perception of the school

zDataPerc = stats.zscore(df_perception_nonan) #normalizing data
pca = PCA()
pca.fit(zDataPerc)

eigValsPerc = pca.explained_variance_

loadingsPerc = pca.components_

rotatedData = pca.fit_transform(zDataPerc) 

covarExplained = eigValsPerc/sum(eigValsPerc)*100
numVariables = 6 

#Plotting the Screeplot and showing the Kaiser criterion (y=1)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.axhline(y = 1, color = 'r')
plt.bar(np.linspace(1,numVariables,numVariables),eigValsPerc)
plt.title("Eigenvalues for students perception (Col L-Q)")

#DOING PCA for Students achievement

zDataPerf = stats.zscore(df_achievement_nonan)
pca = PCA()
pca.fit(zDataPerf)

eigValsPerf = pca.explained_variance_

loadingsPerf = pca.components_

rotatedData = pca.fit_transform(zDataPerf)

covarExplained = eigValsPerf/sum(eigValsPerc)*100
numVariables = 3

#Reading scores are very important with a loading of -0.8

#Performing the ANOVA
#look at the correlation between student achievement and collaborative teachers

corr_achievement_collab = df_perception_nonan['collaborative_teachers'].corr(df_achievement['student_achievement'])

#For the Anova:
achievement = data['student_achievement']
teachers = data['collaborative_teachers']
trust = data['trust']
reading = data['reading_scores_exceed']
leadership = data['effective_school_leadership']
instruction = data['rigorous_instruction']

postPCAData = pd.concat([achievement,teachers,trust,reading,leadership,instruction],axis=1).dropna()
# postPCAData.columns=newNames

r = np.corrcoef(postPCAData,rowvar=False)
plt.imshow(r) 
plt.colorbar()

corrMatrix = postPCAData.corr()
#Here we look all the above variables as main effects, before then combining them all into one big ANOVA in an attempt to
#grasp the interaction effects between them.
model1 = ols('student_achievement ~ (collaborative_teachers) + (trust) + (reading_scores_exceed) + (effective_school_leadership)\
             + (rigorous_instruction) + (collaborative_teachers):(trust) + (collaborative_teachers):(reading_scores_exceed) + (collaborative_teachers):(effective_school_leadership)\
                      + (collaborative_teachers):(rigorous_instruction) + (trust):(reading_scores_exceed)\
                          + (trust):(effective_school_leadership) + (trust):(collaborative_teachers)\
                              + (trust):(rigorous_instruction) + (reading_scores_exceed):(effective_school_leadership)\
                                  + (reading_scores_exceed):(collaborative_teachers) + (reading_scores_exceed):(rigorous_instruction)\
                                      + (effective_school_leadership):(collaborative_teachers) + (effective_school_leadership):(rigorous_instruction) + \
                                          (collaborative_teachers):(rigorous_instruction)',data=postPCAData).fit()
anova_table1 = sm.stats.anova_lm(model1, typ=2)
print(anova_table1)

#only one significant P-value and a high residual sum_sq so we'll try with only the largest aboslut loading values

smallPostPCAData = postPCAData.copy()
smallPostPCAData = smallPostPCAData.drop(['rigorous_instruction','effective_school_leadership'],axis=1)

model1 = ols('student_achievement ~ (collaborative_teachers) + (trust) + (reading_scores_exceed) + (collaborative_teachers):(trust) + (collaborative_teachers):(reading_scores_exceed) +\
                       (trust):(reading_scores_exceed) + (trust):(collaborative_teachers) + \
                           (reading_scores_exceed):(collaborative_teachers)',data=smallPostPCAData).fit()
anova_table2 = sm.stats.anova_lm(model1, typ=2)
print(anova_table2)

#We find something more intersting here, teachers combined with trust aswell as trust reading scores exceed is important on student achievement.

```
