# ML-bagging-and-boosting-methods
Random forest , Adaboost , HMM and Autoencoder 
This module runs us through the advanced process of ml categorising like applications of bagging and boosting . Random forest is most used predictor due to its multiple method use . Encoders are usually used for image recognition 

## Random Forest 
Random Forest is a learning method based on classification of given data and regression of it . As the name suggests , Random Forest consists of multiple trees that tend to end up choosing one output that’s dominantly expected from all trees. 
Random forest is a significant improvement on decision tree as Decision Tree algorithm is vulnerable to a lot of variance but in case of random forest , the best of all the out puts are choose decreasing the error rate . Random tree  works o n the loop holes of decision tree , like decision trees show a significant change in result in case of a small change in data . Random forest  lets each individual tree to come to their own conclusiosn with small changes or replacements in the data set  . this process is known as bagging , the primary method used in random forest . Random forest takes bagging one step ahead using feature bagging by selecting the most dominant features in each case . 

## Hidden Markov Model 
Markov Mode (MM) and Hidden Markov Model (HMM) are made to handle data that can be represented as a sequence of observations with different states . For understanding HMM flow chart we must understand the basic elements of the sequence it consists of 
Initial state , Next state , state transition probability and output probabilities . These are the important four components of MM sequence . The additional state in HMM is the hidden state  

Markov models work on the statement ‘ If this happens what is the possibility for another event also to occur ‘ . Representing this in a diagram as below . 

When we cant observe the state itself but only the results of few of the states , we tend to use HMM . It’s a standard Markov Model with hidden states . 

## Autoencoder 
Autoencoder is a machine learning technique used for predicting unsupervised learning . The process of re encoding the input data is used by the machine to learn the expected outputs from the input . An autoencoder can be used for anomaly detection in data like ECG’s or noise particles in signals or overlapping images and filter out the required images . 

An autoencoder has broadly two important facts , an encoder family and a decoder family . It has three layers  input layer , hidden layer and output layer . 

An ideal autoencoder has the below two properties 
-	Sensitivity towards the input to accurately rebuild 
-	Insensitive enough to not memories or over fit the data prediction 

An autoencoder , when given an image , tends to decreases the dimensionality and compress the data . Then it decodes the above decreased dimensional input back to its original 

## Adaboost 
AdaBoost is a learning method that was created to better the binary classifier . AdaBoost learns from its previous method and continues to correct them making it an iterative process. .AdaBoost is a break of Adaptive Boosting . As it learns from the mistakes of previous weak classifiers , the this becomes a machine learning classifier and the basic classifier it usually uses is decision tree. 

Boosting is a linear regression where the second input is usually the results of a previous weak classifier . 

We have three kinds of AdaBoost methods . 
Real AdaBoost 
Logit AdaBoost 
Gentle AdaBoost 

## Code 
Most of the codes for ml follow the below steps 
- pre processing the data 
- dividing it into training and testing data set
- applying to the model 
- predicting the value 
- error understanding 

## Conclusion 
Adaboost has the least error rate but it takes a lot of time . Random forest works on the many decision trees giving accurate results and using majority voting . 
Encoders and decoders can be used to re create source codes and images 

## References 
https://inst.eecs.berkeley.edu/~cs188/sp08/projects/hmm/project_hmm.html
https://github.com/timzhang642/HMM-Weather/blob/master/Hidden%20Markov%20Model.ipynb
https://waterprogramming.wordpress.com/2018/07/03/fitting-hidden-markov-models-part-ii-sample-python-script/
https://www.annytab.com/hidden-markov-model-in-python/
https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
https://github.com/prosperityai/random-forest/blob/master/random-forest.ipynb
https://www.youtube.com/watch?v=zP1mBAJQNX0
https://machinelearningmastery.com/implement-random-forest-scratch-python/
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
https://learn-us-east-1-prod-fleet02-xythos.content.blackboardcdn.com/5e00ea752296c/12322840?X-Blackboard-Expiration=1669809600000&X-Blackboard-Signature=nlf%2FXIrY4E8Tc6ZYgoQFoUTY6odtmQ4nk3FWe5HnNmQ%3D&X-Blackboard-Client-Id=100310&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27random_forests.pdf&response-content-type=application%2Fpdf&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFYaCXVzLWVhc3QtMSJGMEQCIAUtdeaUfUdbBi4jjh06%2B9g9lkVdfhs9MVTAKTuT0gylAiAJf8hXizIT7ZSmgyP5d1DeW67GLyljnZSJtWG52jKLPCrMBAhvEAIaDDYzNTU2NzkyNDE4MyIM%2Be132F8XR865WNSWKqkEgxAoKn8CizMjzZt%2Ff6%2Fb3CuNM5alt%2Fwc%2F9CU9QCSrjR%2FNecX7gN0kdmERy65qANvtXn667m5p8RomEftEcXP%2BNZEW448tqvKGJ84PsrSx6OhFoSTUxZLL4xLfslvUtlRtZ8hvsTK8YB3zW9Soj5rGisxuqT%2FxeeieloCMuVVdFEMJJZL3n%2B680yAOKOhuemjF4ukqsWBOI9FrohYj22Ob1FjndjLlCtPdrXw8fek7WRRbJKtu7w6Ig9moYPOON1YSxXGF07hzr5vwD2juofbcID3H93I5GhekKeUFALvYncQpca42BX2IkieB5kALGU%2BZe0kLgZDfU1BfqDsgg8ENcjr%2Fzo8plosHF7nbMuA2bdN73PHR8gwiDs%2FKSd8n9ajD6vpytbJLp1LL0lemXHMjOQJ4TflN3Ys%2FXf08rpCSdeR9u4iZhKAc47m6cRs2fw8clZpZGM%2B93H33m1SyePGAdK12jBhNDbn9CsmLgQreNebHR430%2FZEJxl%2BmAaamA11nNEQDVlnZAdz78PKcrEe3KhB7k0FJa%2FcYzuhF2tF3q9lKFc%2FKtXf6KkNPDbEbQtunSdQnZig2hxgTf9o4s%2FGJlpRBHob0GAOy4e4FSHhLc9me7VHXo30HWx%2BDZeAEwX56bLAqZ%2FPPpSHj5E81JL%2Fp750Jwx2PBvZi3zlqgHBnZlQn2w3OuP4gWaTxfsBPY37SOeGor9lTgyJsdiRJdgPoH%2FdvLDblfoo0TCv4JucBjqqAQNJmgP8wLu6KulqMSN2yzdGLrQE8Fw5k%2FkRqWcYftL6pp8WlpUiHQojUTXX6xv%2FgbYk%2FEamLIRyk20wYRcUmVztK6b1O%2FWcz80nKoBYKTy%2Fp585MGCjpMeD3na2Xd0cjL9vm4BqjTlXs6yZPgGkp0AoEuzFizbAUkSVxLYKoO%2BokDBPTVcliMpjetgaJPX0u5NX2V3y1bDN7EHSFhOC78E1AP6MQWHnMZZQ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221130T060000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PLXR5HPR4Y%2F20221130%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=74aa80912eefe8f0b42accf739f861b6f70d07dda1791b5266a990489aec1384



