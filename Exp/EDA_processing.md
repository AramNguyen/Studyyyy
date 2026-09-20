&nbsp;

1. **Check the data**  
- Have a quick look into the data set and identify the missing data  
+ Calculate the data distribution → by that, detect outlier  
+ Calculate and determine the size of the sample  
2. **Processing data:**  
- Filling the gaps, identify the type of the blanks, if the type is:  
+ Numbers → can be filled by the mean() or med()  
+ Objects or String → locate the column we want to clean, identify the method, most common is fill those columns with “Unknown” or “Missing”

  The syntax:

  df\[‘colum’\] \= df\['column'\].fillna('Unknown')

- The method we use to cope with missing objects value:&nbsp;  
+ Step 1: Statistic to get the most \- second \- third common attribute  
+ Step 2 : Overwrite the rest of attributes by “Others” and fill the missing gap by “Unknown”

  By that way, whenever we use the one-hot encoding method, we aren't afraid of the high dimensionality leads to the less accurate model prediction.

3. **Data visualization:**  
- Create graphs for a better view \- deep knowing about the data&nbsp;  
- Discover more relationship between variables  
4. **Formulate hypotheses**  
- To find out the most efficient way for train machine learning model

&nbsp;

  &nbsp;