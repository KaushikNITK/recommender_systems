# Recommender_systems

This project explores various techniques used in the field of recommendation, including collaborative filtering, content-based methods, hybrid approaches, and advanced deep learning-based solutions like Two Tower architecture

### Inference or Setup instructions
this project is based on python 3.11 and above
to download python 3.11 go to https://www.python.org/downloads/

To create a envi to install the required libraries(If you have an envi activate it and go to step 3)

1. navigate to the location where you want to create your envi by using
```bash
cd ./path_to_your_loc
```

2. Activate the envi by using
 ```bash
.\env\Scripts\activate
```
3. download this repository and store in the location

4. Install requirements.txt by downloading and using this command
 ```bash
pip install requirements.txt
```

5. use the following list of commands to inference
```bash
cd ./recommender_systems/src
```
```bash
uvicorn inference:app --reload --host 0.0.0.0 --port 8000
```

now wait until you see loading succeessful 
##### your API is ready to use
### Loading API
1. Only username as input, url should look like http://localhost:8000/feed?username=your_username
2. both username and catogary_id as input, url should look like http://localhost:8000/feed?username=your_username&category_id=category_id_user_want_to_see
3. all three (username, catogary_id and mood(emoji or text)) as input, url should look like http://localhost:8000/feed?username=your_username&category_id=category_id_user_want_to_see&mood=user_current_mood
replace user_name, category_id_user_want_to_see and user_current_mood with appropriate vinputs
### Data cleaning and Model
This Recommender systems is built with two tower architecture with multiple optimizations like post embeddings saving, user to all post embeddings. Data cleaning is done by converting the data given in the api to pandas dataframe, exraction of data, post summary embeddings by bert etc.,
#### data cleaning and processing
Data is extracted form api, converted to pandas dataframe and interaction scores are created based on the strength of interaction between the user and each post. All post details are cleaned and extracted using bert embedings of post summary given. Category id is extacted and used while inferencing.

#### Model 
Model architecture includes two towers one encodes user information called user tower, similerly there was post tower which is unnecesary as all the posts are fixed(in case of refreshing posts we can seperatly compute and use them here) so the post tower is trained and post embedings are taken out from tha post tower and stored in post_embeddings.csv. these post embeddings are used later in the model's inferencing.
advantages of two tower over others are 
* it provides relation of user with all valid posts and then filter the top recommendations
* complex and powerful model
* can provide better recomendations even on cold start

Model evaluation on given dataset
on 10 epochs
under MSE loss training-0.015, validation loss 0.045

##### insights
As the posts are fixed i have seperatly stored all posts embedings. this embeddings are directly used to find similarities with user embeddings

While inferencing we have three catogaries filtering of posts happens as follows
1. Only user name - For this all post's embeddings are given to model
2. user name and catogary - For this post embeddings with post's category id same as the input category id are given to model
3. all three are given - for this initially post embeddings are arraged in the order of similarity with given mood input(mood input is embedded by bert and taken similarity with post embedings and arranged) then rigidly seperated by category id then the final post embeddings are sent to the mode.

 ### Output
 Final output of the model
 these are the possible cases
 1. Given user name then the output is list of 10 video links with its post id
 2. Given user name and category id then the output can be a list of 10 or less than 10 video links with post id
 3. Given all three it is same as case 2
 4. In case user name does not exist in the database and given mood and category id then output will be same as case 2
 5. In case user name does not exist in the database and has not mood OR category id then output will be an error with status code 404 User is not in database so give input of mood and category_id
 6. In case of invalid category id the output will be error with status code 404 invalid category id
 7. incase of invalid server error there might be error in the inference code
