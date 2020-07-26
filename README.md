# Movie Recommendation System
A DNN for recommending movies based on the user's rating history. 

## Model

The model takes a user's rating history as the input and uses an autoencoder architecture to predict the user's rating for all movies in the data. The input is a list with size equal to the number of movies consisting of the user's previous ratings excluding one (0 for the Not-Rated and withheld movies). The output is also a list of the same size but could be either just the withheld rating or the complete rating history. The loss is calculated such that it only considers the ratings in the output of the example (y_true) and the corresponding predicted ratings (y_pred). The number and size of the layers in the model depend on the number of movies so that more similarities and differences between the movies can be learned by the model. 

## Prediction

As the requirement is not to predict the ratings but recommend movies to the user, the predicted ratings are used to generate a list for each user which recommends movies that the user may like.

## Results

The model trained using the complete rating history output performed better compared to the one using the single withheld output with 10x smaller loss values. Both models converged to the least validation error in the 28<sup>th</sup> epoch. More details in Results.xlsx.

| Output Format   | Train | Validation | Test  |
|-----------------|-------|------------|-------|
| Full History    | 0.069 | 0.066      | 0.066 |
| Single Withheld | 0.607 | 0.738      | 0.739 |

## Reference Paper

[Movie Recommendations Using the Deep Learning Approach](https://ieeexplore.ieee.org/document/8424686)
