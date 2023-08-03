# NASA Close Objects Logistic Regression
## Skills Used: Python (pandas, numpy, sklearn LogisticRegression), Tableau
### Data From NASA's [Close Earth Objects Dataset]([https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects))
Personal Project classifying asteroids and other space objects as either hazardous or not hazardous based on the explanatory variabels given in the dataset.  Began with EDA and used NASA's documentation to define columns of interest.  Converted all necessary data to float or int type.  Created X and y variables for LogisticRegression and fit model.  Set random_state to 50 to ensure reproductability.  Exported data to Tableau for data visualization.  Dashboard can be found [here](https://public.tableau.com/app/profile/alberto.rivas.ii)

## Posibilities for Further Research
- Fine tune model to account for data skewness (81996 False values, 8840 True values), try penalty parameter
- Look at correlations between explanatory variables to look for multicolinearity
