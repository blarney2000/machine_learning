<div class="container-fluid main-container">


<div id="header">
<h1 class="title">Machine Learning Prediction Writeup</h1>
</div>


<div id="human-activity-recognition-with-wearable-accelerometers" class="section level2">
<h2>Human Activity Recognition with Wearable Accelerometers</h2>
<div id="data-description" class="section level3">
<h3>Data Description</h3>
<p>Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now easy to collect a large amount of data about personal activity. These types of devices are part of the “quantified self movement - a group of enthusiasts who take measurements of their actions regularly. Although some people often quantify how much of a particular activity they do, they rarely quantify how well they do it. This project uses data from accelerometers on the belt, forearm, arm, and dumbell of six participants. They were asked to perform barbell lifts correctly and incorrectly in five different ways:</p>
<pre><code>1. Exactly according to the specification
2. Throwing elbows to the front
3. Lifting the dumbbell only halfway
4. lowering the dumbbell only halfway
5. throwing the hips to the front</code></pre>
<p>More information is available from the website here: <a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a> (see the section on the Weight Lifting Exercise Dataset).</p>
</div>
<div id="objectives" class="section level3">
<h3>Objectives</h3>
<p>The goal of this project was to predict the manner (see above) in which the subjects did the exercise when given relevant information.</p>
</div>
<div id="preparing-the-data" class="section level3">
<h3>Preparing the Data</h3>
<p>The data was obtained from: <a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv</a></p>
<p>The data source has 159 variables. The first seven columns of the data (<i>X, user name, raw timestamp part 1, raw timestamp part 2, cvtd timestamp, new window, num window</i>) will be removed, since these do not appear to be accelerometer measurements which will not have an effect in prediction.</p>
<p>Sparse variables have few observations, so they have weak predictive value. These variables will be removed unless at least 80% of their observations are present.</p>
<p>Variables with few unique values are also removed, since their invariance adds little to the predictive accuracy of the model.</p>
<p>Lastly, variables that are too highly correlated are removed since they are not very useful. This step reduced the number of variables in the data from 53 to 46.</p>
<pre class="r"><code>dim(data.training)</code></pre>
<pre><code>## [1] 19622   159</code></pre>
<pre class="r"><code>data.training &lt;- data.training[,c(8:ncol(data.training))] #from 159 vars

dim(data.training)</code></pre>
<pre><code>## [1] 19622   152</code></pre>
<pre class="r"><code>data.training &lt;- data.training[,colSums(is.na(data.training)) &lt; .8]

dim(data.training)</code></pre>
<pre><code>## [1] 19622   119</code></pre>
<pre class="r"><code>nsv &lt;- nearZeroVar(data.training, saveMetrics = TRUE)
data.training &lt;- data.training[,!nsv$nzv]

dim(data.training)</code></pre>
<pre><code>## [1] 19622    52</code></pre>
<pre class="r"><code>highCorrelations &lt;- cor(na.omit(data.training[sapply(data.training, is.numeric)]))
highCorr&lt;-findCorrelation(highCorrelations, cutoff = .90, verbose = FALSE)
data.training&lt;- data.training[,-highCorr]

dim(data.training)</code></pre>
<pre><code>## [1] 19622    46</code></pre>
<p>This leaves us with 45 predictors and the “classe” variable in the training data.</p>
<p>We will split the training data into a “training” set and a “validating” set. The training subset will include 75% of the observations and the remaining validation data will be used to calculate the out-of-sample error.</p>
<pre class="r"><code>xdata &lt;- createDataPartition(y = data.training$classe, p = .75, list = FALSE )
data.validating &lt;- data.training[-xdata,]
data.training &lt;- data.training[xdata,]</code></pre>
<div id="random-forest-model" class="section level4">
<h4>Random Forest Model</h4>
<p>There does not seem to be any predictors strongly correlated with the outcome variable, so linear regression model may not be a good option. Instead, a classifcation algorithm – Random Forest model – will be used. This model uses bagging and random subsets of variables from the training data to prevent overfitting. In this algorithm, many different trees are created for cross validation.</p>
<pre class="r"><code>model.rf &lt;- randomForest(classe~., data = data.training)</code></pre>
</div>
<div id="training-statistics" class="section level4">
<h4>Training Statistics</h4>
<pre class="r"><code>predict.rf &lt;- predict(model.rf, data.training, type = &quot;class&quot;)
confusionMatrix(predict.rf, data.training$classe)</code></pre>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000</code></pre>
<p>Now we test our model on the out-of-sample dataset. The error is expected to be higher than with our out-of-sample data, but hopefully as close as possible.</p>
</div>
<div id="expected-out-of-sample-error" class="section level4">
<h4>Expected Out-of-Sample Error</h4>
<p>The model was tested on the validation data to get the expected out-of-sample error.</p>
<pre class="r"><code>predictions.validating &lt;- predict(model.rf, newdata = data.validating)
confusionMatrix(predictions.validating, data.validating$classe)</code></pre>
<pre><code>## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    4    0    0    0
##          B    0  938    7    0    0
##          C    0    7  847    7    1
##          D    0    0    1  796    0
##          E    1    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc &gt; NIR] : &lt; 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9884   0.9906   0.9900   0.9989
## Specificity            0.9989   0.9982   0.9963   0.9998   0.9995
## Pos Pred Value         0.9971   0.9926   0.9826   0.9987   0.9978
## Neg Pred Value         0.9997   0.9972   0.9980   0.9981   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1913   0.1727   0.1623   0.1835
## Detection Prevalence   0.2851   0.1927   0.1758   0.1625   0.1839
## Balanced Accuracy      0.9991   0.9933   0.9935   0.9949   0.9992</code></pre>
<p>Calculate the the expected out-of-sample error:</p>
<pre class="r"><code>oose &lt;- 1 - as.numeric(confusionMatrix(predictions.validating, data.validating$classe)$overall[1])
oose</code></pre>
<pre><code>## [1] 0.00591354</code></pre>
<p>Our Random Forest algorithm generates a model with accuracy 0.994 on our validation data. The out-of-sampe error is 0.59%. Since this is satisfactory, there is no need to go back and include more variables with imputations. However, we will still look at which variables are the most important.</p>
<pre class="r"><code>varImp(model.rf, scale = TRUE)</code></pre>
<pre><code>##                        Overall
## pitch_belt           614.51519
## yaw_belt             826.64985
## total_accel_belt     278.27631
## gyros_belt_x          99.36361
## gyros_belt_y         126.55245
## gyros_belt_z         331.28747
## magnet_belt_x        227.95236
## magnet_belt_y        412.54980
## magnet_belt_z        412.10509
## roll_arm             282.20661
## pitch_arm            144.35650
## yaw_arm              194.87720
## total_accel_arm       90.87674
## gyros_arm_x          134.50260
## gyros_arm_z           57.43171
## accel_arm_x          195.56975
## accel_arm_y          145.96397
## accel_arm_z          124.56280
## magnet_arm_x         209.18897
## magnet_arm_y         186.65258
## magnet_arm_z         162.54059
## roll_dumbbell        333.58729
## pitch_dumbbell       153.25792
## yaw_dumbbell         225.44459
## total_accel_dumbbell 222.36165
## gyros_dumbbell_y     221.46724
## gyros_dumbbell_z      81.74818
## accel_dumbbell_x     207.15664
## accel_dumbbell_y     324.12228
## accel_dumbbell_z     283.45309
## magnet_dumbbell_x    391.91952
## magnet_dumbbell_y    549.68404
## magnet_dumbbell_z    623.31366
## roll_forearm         497.15960
## pitch_forearm        607.12650
## yaw_forearm          149.73222
## total_accel_forearm  100.91705
## gyros_forearm_x       74.26957
## gyros_forearm_y      120.54555
## accel_forearm_x      257.38150
## accel_forearm_y      120.79011
## accel_forearm_z      214.52282
## magnet_forearm_x     186.94267
## magnet_forearm_y     190.61587
## magnet_forearm_z     240.44768</code></pre>
<p>The four most important variables according to the model fit are ‘yaw_belt’, ‘pitch_forearm’, ‘magnet_dumbell_z’ and ‘pitch_belt’. These variables have the most effect on the model’s predictive performance.</p>
<p>The model could be fine-tuned to use only the most important predictors. Although this could increase the speed of the model without reducing accuracy significantly, this step is not needed for this project.</p>
</div>
<div id="conclusion" class="section level4">
<h4>Conclusion</h4>
<p>A Random Forest model was created to predict the manner that an exercise was performed given relevant data predicting five classes (‘A’, ‘B’, ‘C’, ‘D’, ‘E’) using 45 predictors. The model We used 53 variables from the training dataset to build a random forest model with four-fold cross validation. The accuracy of the model is 0.988 tested on the out-of-sample data above.</p>
</div>
</div>
</div>


</div>

<script>

// add bootstrap table styles to pandoc tables
$(document).ready(function () {
  $('tr.header').parent('thead').parent('table').addClass('table table-condensed');
});

</script>

<!-- dynamically load mathjax for compatibility with self-contained -->
<script>
  (function () {
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.src  = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
    document.getElementsByTagName("head")[0].appendChild(script);
  })();
</script>

</body>
</html>
