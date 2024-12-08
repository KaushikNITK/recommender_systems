# Recommender_systems

This project explores various techniques used in the field of recommendation, including collaborative filtering, content-based methods, hybrid approaches, and advanced deep learning-based solutions like Two Tower architecture

### Inferenceing

create a envi to install the required libraries
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
##### your api is ready to use

### Data cleaning and Model
This Recommender systems is built with two tower architecture with multiple optimizations like post embeddings saving, user to all post embeddings. Data cleaning is done by converting the data given in the api to pandas dataframe, exraction of data, post summary embeddings by bert etc.,
##### data cleaning and processing
Data is extracted form api, converted to pandas dataframe and interaction scores are created based on the strength of interaction between the user and each post. All post details are cleaned and extracted using bert embedings of post summary given. Category id is extacted and used while inferencing.

#### Model 
Model architecture includes two towers one encodes user information called user tower, similerly there was post tower which is unnecesary as all the posts are fixed so the post tower is trained and post embedings are taken out from tha post tower and stored in post_embeddings.csv. these post embeddings are used later in the model's inferencing.

While inferencing we have three catogaries 
1. Only user name - For this all post's embeddings are given to model
2. user name and catogary - For this post embeddings with post's category id same as the input category id are given to model
3. all three are given - for this initially post embeddings are arraged in the order of similarity with given mood input(mood input is embedded by bert and taken similarity with post embedings and arranged) then rigidly seperated by category id then the final post embeddings are sent to the mode.

 ### Output
 Final output of the model
 these are the possible cases
 1. Given user name then the output is list of 10 video links with its post id
 2. Given user name and category id then the output can be a list of 10 or less than 10 video links with post id
 3. Given all three it is same as case 2
 4. In case user name does not exist in the database output is an error with status code 404 User not found
 5. In case of invalid category id the output will be error with status code 404 invalid category id
 6. incase of invalid server error there might be error in the inference code
