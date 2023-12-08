# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
When I commented out the standard scaler the accuracy decreased to 0.73 which is not too accurate. An ideal accuracy score would at least be 0.80 but since it is below this threshold it makes it unaccurate. This is because since there isn't a scaler it make so that there isn't a interval range. This makes the data change because there might be a value that's an outlier which can make it harder for it to be more accurate.
2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
The accuracy with the standard scaler is better as it goes up to 0.88. Because the higher value is 1, I think .88 is accurate enough because it is close to the 1 value. 
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
The model did really well because almost all the predicted values are te actual values. There was a pattern in the model predicting more men when in reality the actual genders were female. We can see that almost all of the incorrect matches defaulted the predicted gender in being female.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
According to my model a 34 year old female who makes 56k a year would  buy an SUV because when I ran the program it printed a 1 which means that yes they would likely buy the suv.