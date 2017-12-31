# HoopNet
A machine learning model that predicts win probabilities for NBA games.

### The Data
The model is trained on historical data scraped from various sources.  The data features include advanced team statistics (mostly efficiency-type statistics) and spatial tracking data (such as shooting percentages from various spots on the floor).  
To obtain the data, run "nba_stat_scraper.py" with the for loop running over all years (2002-2018).  This will construct two csv files per year: a table of each team's statistics for that year and a table of each game played that year with the home team's data featured first (the "x" columns) and the visiting team's statistics last.  The "winner" column in the games table denotes the winner of the game ("0" = home, "1" = visitor).

### The model(s)
The historical data is then used to train several models: a logistic regression model, a random forest, a gradient-boosted model and a deep neural network.  To train the first three models, run "hoopNet.py" and to train the neural net run "cnnHoopNet.py".  (Note: also included in the files is an attempt to train an autoencoder neural net with a logistic regression classifier attached at the end ("aeHoopNet.py").  No significant improvment in performance was observed.)

Of all the models trained, the logistic regression and the convolutional neural net achieve the highest performance on test data (~69% accuracy which is supposedly on par with human predictors).

### Daily Predictions
Finally, run "daily_predictions.py" to obtain the predictions for the NBA games of the current day.  (Note: to keep the data up-to-date, first run "nba_stat_scraper.py" for the year 2018 only.  This will update the team statistics for the current year and ensure the most precise prediction possible.)

The output will show the predictions of each model and an overall average.
