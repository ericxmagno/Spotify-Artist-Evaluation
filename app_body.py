import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
from streamlit import caching
import streamlit.components.v1 as components

def objectives():
    st.write('')
    st.header('Objectives')
    st.write('-----------------------------------------------------------------------') 
    st.subheader('- To determine which song features Lola Amour can focus on to improve their chances to penetrate the daily Spotify Top 200 tracks in the Philippines')
    st.write('')
    st.subheader('- To suggest which artist(s) Lola Amour should collaborate with for their next release')
    

def client_profile():
    caching.clear_cache()
    st.write('')
    st.header('Client Profile')
    st.write('-----------------------------------------------------------------------') 
    st.subheader('About the Artist')
    st.write('')
    col1a, col2a = st.beta_columns(2)
    with col1a:
        image = Image.open('images/artist.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=300)
        components.html(
            """
            <iframe src="https://open.spotify.com/follow/1/?uri=spotify:artist:29zSTMejPhY0m7kwNQ9SPI&size=detail&theme=light" width="300" height="56" scrolling="no" frameborder="0" style="border:none; overflow:hidden;" allowtransparency="true"></iframe>        
            """
        )
        st.write('')
        st.subheader('Most Popular Songs')
        st.write('1. Pwede Ba (9,400,987 streams)')
        st.write('2. Sundan Mo Ko (2,706,107 streams)')
        st.write('3. Maybe Maybe (2,699,013 streams)')
    with col2a:
        st.write('Lola Amour is a 7-piece band that dabbles in the genres of modern rock, funk, and pop. Their peculiar sound'+
        ' is a synthesis of disparate musical influences. The band began creeping its way slowly to the local Philippine music scene in 2016'+
        ' and has since then gaining traction with their music and whimsical personalities.')
        st.write('Since then they have released their debut EP "Don\'t Look Back" and their single "Pwede Ba". The band is currently working on their first full-length album.')
        st.markdown("<h5 style='text-align: right;'>- Taken from Spotify</h5>", unsafe_allow_html=True)
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.subheader('Years Active: 2013 - present')
        st.subheader('Genres: Pinoy Indie, Pinoy Rock')

    st.write('')
    
    st.write('-----------------------------------------------------------------------') 
    col1b, col2b = st.beta_columns(2)
    with col1b:
        st.subheader('Popular Tracks')
        st.write('')
        components.html(
            """
            <iframe src="https://open.spotify.com/embed/artist/29zSTMejPhY0m7kwNQ9SPI" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
            """
        )
    with col2b:
        st.subheader('Artist Radio')
        st.write('')
        components.html(
            """
            <iframe src="https://open.spotify.com/embed/playlist/37i9dQZF1E4m0JRPlwKfpC" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
            """
        )
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    col1c, col2c = st.beta_columns(2)
    with col1c:
        st.subheader('Spotify Statistics')
        st.text('as of 02/05/2021')
        statistics = {
                          'Info': ['Monthly Listeners', 'Followers'], 
                          '': ['225,216','75,659']}
        st.table(pd.DataFrame(statistics).set_index('Info'))
    with col2c:
        st.subheader('Where People Listen')
        st.text('as of 02/05/2021')
        location = {
                          'City': ['Quezon City, PH', 'Caloocan City, PH', 'Makati City, PH', 'Manila, PH', 'Roosevelt, PH'], 
                          'Listeners': ['31,682','16,073','13,264','12,861','10,233']}
        st.table(pd.DataFrame(location).set_index('City'))
        

def dataset():
    st.write('')
    st.header('Data Information')
    st.write('-----------------------------------------------------------------------') 
    st.write('This project is based on data scraped from Spotify.')
    st.text("")

    st.header('Data Sets:')
    st.write('')

    st.subheader('Audio Features of Lola Amour\'s Top Tracks')
    df_artist_data = pd.read_csv('data/artist_data_lola_amour.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Top Pinoy Indie Playlists')
    df_artist_data = pd.read_csv('data/pinoy indie_playlist_data.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Audio Features of Pinoy Indie Tracks')
    df_artist_data = pd.read_csv('data/pinoy indie_playlist_tracks_data.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Top Pinoy Rock Playlists')
    df_artist_data = pd.read_csv('data/pinoy rock_playlist_data.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Audio Features of Pinoy Rock Tracks')
    df_artist_data = pd.read_csv('data/pinoy rock_playlist_tracks_data.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Spotify Daily Charts')
    df_artist_data = pd.read_csv('data/spotify_daily_charts.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Spotify Daily Charts Artists')
    df_artist_data = pd.read_csv('data/spotify_daily_charts_artists.csv')
    st.write(df_artist_data.head(10))
    st.write('')

    st.subheader('Spotify Daily Charts Track Features')
    df_artist_data = pd.read_csv('data/spotify_daily_charts_tracks.csv')
    st.write(df_artist_data.head(10))
    st.write('')
    # st.markdown('<b>Feature Set:</b>', unsafe_allow_html=True)
    # featureset = {
    #                   'Column Name': ['school.classification', 'Schools Location', 'Number of Rooms', 'Number of Teachers', 'Enrollment Master', 'MOOE'], 
    #                   'Rows': ['46,603', '46,624', '46,412', '45,040', '46,626', '46,028'],
    #                   'Columns': ['22', '12', '5', '5', '17', '5'],
    #                   'Description': ['Masterlist of Public Elementary and Secondary Schools', 'Location of Public Schools', 'Instructional Rooms in Public Elementary and Secondary Schools', 'Masterlist of Public School Teachers', 'Total Enrollment in Public Elementary and Secondary Schools', 'Maintenance and Other Operational Expenses (MOOE) allocation for Public Elementary and Secondary Schools']
    #                  }
    # st.table(featureset)

def tools():
    st.write('')
    st.header('List of Tools')
    st.write('-----------------------------------------------------------------------') 
    image = Image.open('logo/jupyter.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/pandas.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/heroku.jpg').convert('RGB')
    st.image(image, caption='', width=150, height=50)
    image = Image.open('logo/streamlit.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/github.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/scipy.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/seaborn.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/matplotlib.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)
    image = Image.open('logo/numpy.png').convert('RGB')
    st.image(image, caption='', width=300, height=150)


def eda1():
    caching.clear_cache()
    st.write('')
    st.header('EDA - About the Industry')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Quick Facts')
    st.write('')
    st.text('113 - Number of Pinoy Rock and Pinoy Indie songs that made it into the Spotify\ndaily top 200 charts in the Philippines from 2017-present.')
    st.text('3 - Songs that made the #1 spot. \n(Buwan by Juan Karlos, Mundo by IV of Spades, Sana by by I Belong to the Zoo.)')
    st.text('3.44% - Percentage of the songs in the charts that are either Pinoy Rock or Pinoy Indie.')
    st.text('0 - Songs of Lola Amour have made the top 200 charts so far.')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader("Total Streams of Related Genres")
    st.write('')
    image = Image.open('figures/eda_1.png').convert('RGB')
    st.image(image, caption='', width=900, height=600)
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader("Most Popular Artists for Pinoy Indie")
    st.write('')
    col1a, col2a = st.beta_columns(2)
    with col1a:
        st.subheader('1. IV of Spades')
        st.write('No. of times in top 200: 1,604')
        st.write('Total Steams: 108,874,780')
    with col2a:
        image = Image.open('images/4os.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    col1b, col2b = st.beta_columns(2)
    with col1b:
        st.subheader('2. Up Dharma Down')
        st.write('No. of times in top 200: 1,562')
        st.write('Total Steams: 77,975,277')
    with col2b:
        image = Image.open('images/udd.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    col1c, col2c = st.beta_columns(2)
    with col1c:
        st.subheader('3. I Belong to the Zoo')
        st.write('No. of times in top 200: 1,518')
        st.write('Total Steams: 155,143,287')
    with col2c:
        image = Image.open('images/zoo.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader("Most Popular Artists for Pinoy Rock")
    st.write('')
    col1d, col2d = st.beta_columns(2)
    with col1d:
        st.subheader('1. Silent Sanctuary')
        st.write('No. of times in top 200: 3,800')
        st.write('Total Steams: 136,866,048')
    with col2d:
        image = Image.open('images/ss.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    col1e, col2e = st.beta_columns(2)
    with col1e:
        st.subheader('2. Hale')
        st.write('No. of times in top 200: 3,033')
        st.write('Total Steams: 104,943,404')
    with col2e:
        image = Image.open('images/hale.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    col1f, col2f = st.beta_columns(2)
    with col1f:
        st.subheader('3. Callalily')
        st.write('No. of times in top 200: 2,110')
        st.write('Total Steams: 77,851,978')
    with col2f:
        image = Image.open('images/callalily.jpg').convert('RGB')
        st.image(image, caption='', width=300, height=200)

def eda2():
    caching.clear_cache()
    st.write('')
    st.header('EDA - Audio Features')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Audio Feature Comparison: Lola Amour vs Pinoy Indie Playlists')
    st.write('')
    st.subheader('They Differ in:')
    st.write('')
    col1a, col2a = st.beta_columns(2)
    with col1a:
        image = Image.open('images/pinoy_indie_danceability.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    with col2a:
        image = Image.open('images/pinoy_indie_tempo.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    st.write('')
    st.subheader('They Fit in:')
    st.write('')
    col1a, col2a = st.beta_columns(2)
    with col1a:
        image = Image.open('images/pinoy_indie_loudness.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
        image = Image.open('images/pinoy_indie_energy.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
        image = Image.open('images/pinoy_indie_speechiness.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    with col2a:
        image = Image.open('images/pinoy_indie_valence.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
        image = Image.open('images/pinoy_indie_liveness.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Audio Feature Comparison: Lola Amour vs Pinoy Rock Playlists')
    st.write('')
    st.subheader('They Differ in:')
    st.write('')
    col1a, col2a = st.beta_columns(2)
    with col1a:
        image = Image.open('images/pinoy_rock_danceability.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
        image = Image.open('images/pinoy_rock_energy.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    with col2a:
        image = Image.open('images/pinoy_rock_tempo.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    st.write('')
    st.subheader('They Fit in:')
    st.write('')
    col1a, col2a = st.beta_columns(2)
    with col1a:
        image = Image.open('images/pinoy_rock_loudness.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
        image = Image.open('images/pinoy_rock_speechiness.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
    with col2a:
        image = Image.open('images/pinoy_rock_valence.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)
        image = Image.open('images/pinoy_rock_liveness.png').convert('RGB')
        st.image(image, caption='', width=300, height=200)

def modeling():
    caching.clear_cache()
    st.write('')
    st.header('Modeling - Results')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Correlation Matrix of Track Audio Features')
    if st.checkbox('Show code', value=False, key="1"):
        st.code("""
        #get top 10 nearest to seed_track_data
        viz_data = df_indie_rock.drop(columns=['date', 'position', 'track_id', 'track_name', 'artist', 'streams', 'artist_id', 'album_id', 'duration', 'release_date'])
        plt.subplots(figsize=(16, 6))
        sns.heatmap(viz_data.corr(), annot=True)
        """, language="python")
    st.write('')
    image = Image.open('images/corr.png').convert('RGB')
    st.image(image, caption='', width=900, height=600)
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Undersampling to Fix Target Imbalance')
    if st.checkbox('Show code', value=False, key="2"):
        st.code("""
        # create feature matrix (X)
        feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',\
                        'liveness', 'valence', 'tempo']
        X = df_indie_rock[feature_cols]
        y = df_indie_rock['is_top50']

        # summarize class distribution
        counter = Counter(y)
        print(counter)
        # define the undersampling method
        undersample = NearMiss(version=1, n_neighbors_ver3=3)
        # transform the dataset
        X, y = undersample.fit_resample(X, y)
        # summarize the new class distribution
        counter = Counter(y)
        print(counter)
        """, language="python")
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Split Dataset into Train and Test')
    if st.checkbox('Show code', value=False, key="3"):
        st.code("""
        X_train,X_test,Y_train,Y_test = train_test_split(X,y, test_size=0.2)  # 0.2 = 20% of my data set for testing
        print("Shape of X_Train:"+str(X_train.shape))
        print("Shape of y_Train:"+str(Y_train.shape))
        print("Shape of X_Test:"+str(X_test.shape))
        print("Shape of y_Test:"+str(Y_test.shape))
        """, language="python")
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('KNN Model')
    if st.checkbox('Show code', value=False, key="4"):
        st.code("""
        cv_scores = []

        neighbors=np.arange(2,51)

        for k in neighbors:
            print('Fitting for k=%d' % k)
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy')   # cv = k-fold
            cv_scores.append(scores.mean())
            
        # changing to misclassification error
        mse = [1 - x for x in cv_scores]

        # determining best k
        optimal_k = neighbors[mse.index(min(mse))]
        print("The optimal number of neighbors is {}".format(optimal_k))

        # plot misclassification error vs k
        plt.plot(neighbors, mse)
        plt.xlabel("Number of Neighbors K")
        plt.ylabel("Misclassification Error")
        plt.show()

        ####################################################
        knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)

        # fitting the model
        knn_optimal.fit(X_train, Y_train)

        # predict the response
        pred = knn_optimal.predict(X_test)

        # evaluate accuracy
        acc = accuracy_score(Y_test, pred) * 100
        print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, acc))

        #Classification Report
        print(classification_report(Y_test,knn_optimal.predict(X_test)))
        """, language="python")
    st.write('')
    image = Image.open('images/optimal_knn.png').convert('RGB')
    st.image(image, caption='', width=600, height=400)
    st.write('')
    image = Image.open('images/confusion.png').convert('RGB')
    st.image(image, caption='', width=600, height=400)
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Random Forest Model')
    if st.checkbox('Show code', value=False, key="5"):
        st.code("""
        from sklearn.model_selection import RandomizedSearchCV
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        print(random_grid)

        from sklearn.ensemble import RandomForestClassifier
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=10, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        pred = rf_random.predict(X_test)

        from sklearn.metrics import accuracy_score, classification_report
        # evaluate accuracy
        acc = accuracy_score(y_test, pred) * 100
        print('\nThe accuracy of the random forest classifier is %f%%' % (acc))
        print(classification_report(y_test,rf_random.predict(X_test)))
        """, language="python")
    st.write('')
    st.write('The accuracy of the random forest classifier is 93.333333%')
    st.write('')
    image = Image.open('images/confusion2.png').convert('RGB')
    st.image(image, caption='', width=600, height=400)
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Feature Importance')
    st.write('')
    if st.checkbox('Show code', value=False, key="6"):
        st.code("""
        importance = best_model.feature_importances_
        ax = sns.barplot(y = feature_cols, x = importance)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        """, language="python")
    st.write('')
    image = Image.open('images/feature_importance.png').convert('RGB')
    st.image(image, caption='', width=600, height=400)
    st.write('')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    st.subheader('Comparison of Lola Amour\'s Aggregated Sound vs Predicted Top 200 Pinoy Indie/Rock')
    st.write('')
    image = Image.open('images/post_eda_tempo.png').convert('RGB')
    st.image(image, caption='', width=450, height=300)
    st.write('Seeded tracks mean tempo is 120.6 bpm')
    st.write('')
    image = Image.open('images/post_eda_danceability.png').convert('RGB')
    st.image(image, caption='', width=450, height=300)
    st.write('Seeded tracks mean danceability score is 0.55.')
    st.write('')
    image = Image.open('images/post_eda_energy.png').convert('RGB')
    st.image(image, caption='', width=450, height=300)
    st.write('Seeded tracks mean energy score is 0.50.')
    

    
def recommendation_engine():
    caching.clear_cache()
    st.write('')
    st.header('Recommendation Engine')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader("Tracks Most Similar to 'Pwede Ba' (Pinoy Indie)")
    if st.checkbox('Show code', value=False, key="1"):
        st.code("""
        #get top 10 nearest to seed_track_data
        recommendation_df = tracks_df[tracks_df['predicted_genre']=='pinoy indie']\
                                            [tracks_df['track_id']!=seed_track_data['track_id']]\
                                            .sort_values('cosine_dist')[:10]
        recommendation_df[['track_name','artist_name','cosine_dist','predicted_genre']+feature_cols]
        recommendation_df.to_csv('../data/recommended_pinoy_indie.csv')
        """, language="python")
    df_indie = pd.read_csv('data/recommended_pinoy_indie.csv')
    st.write(df_indie.set_index('track_id').drop(columns=['Unnamed: 0']))
    st.write('')
    st.subheader('Top Artists with a Similar "Pinoy Indie" Sound:')
    st.write('Markki Stroem, Brisom, Gabe Bondoc, TALA, Jensen & The Flips')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    

    st.subheader("Tracks Most Similar to 'Pwede Ba' (Pinoy Rock)")
    if st.checkbox('Show code', value=False, key="2"):
        st.code("""
        #get top 10 nearest to seed_track_data
        recommendation_df = tracks_df[tracks_df['predicted_genre']=='pinoy rock']\
                                            [tracks_df['track_id']!=seed_track_data['track_id']]\
                                            .sort_values('cosine_dist')[:10]
        recommendation_df[['track_name','artist_name','cosine_dist','predicted_genre']+feature_cols]
        recommendation_df.to_csv('../data/recommended_pinoy_indie.csv')
        """, language="python")
    df_indie = pd.read_csv('data/recommended_pinoy_rock.csv')
    st.write(df_indie.set_index('track_id').drop(columns=['Unnamed: 0']))
    st.write('')
    st.subheader('Top Artists with a Similar "Pinoy Rock" Sound:')
    st.write('6cyclemind, Bamboo, Shamrock, Cinderell, Fred Engay')


def candr():
    caching.clear_cache()
    st.write('')
    st.header('Conclusions and Recommendations')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Conclusions:')
    st.markdown('- Compose songs that are **higher tempo**, **lower danceability**, and that are **energetic.**')
    st.markdown('- Recommended **Pinoy Indie** artists to collaborate with are **Markki Stroem, Brisom, Gabe Bondoc, TALA, Jensen & The Flips**.')
    st.markdown('- Recommended **Pinoy Rock** artists to collaborate with are **6cyclemind, Bamboo, Shamrock, Cinderell, Fred Engay**.')

    st.write('')

    st.subheader('Recommendations:')
    st.markdown('- Explore different machine learning algorithms for classification.')
    st.markdown('- Take a look at how seasonality affects pinoy indie/rock overall streams.')
    st.markdown('- Explore different evaluation metrics of success other than charting in the Top 200 Philippines playlist.')

def contributors():
    caching.clear_cache()
    st.write('')
    st.header('Contributors')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Edward Nataniel Apostol')
    st.markdown('- Email: [edward.nataniel@gmail.com](edward.nataniel@gmail.com)')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/edward-apostol/](https://www.linkedin.com/in/edward-apostol/)')

    st.subheader('Eric Vincent Magno')
    st.markdown('- Email: [ericvincentmagno@gmail.com](mailto:ericvincentmagno@gmail.com)')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/ericxmagno/](https://www.linkedin.com/in/ericxmagno/)')

    st.subheader('Fatima Grace Santos')
    st.markdown('- Email: [fatima.santos02@yahoo.com](fatima.santos02@yahoo.com)')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/fatima-grace-santos/](https://www.linkedin.com/in/fatima-grace-santos/)')

    st.subheader('Joseph Figuracion')
    st.markdown('- Email: [josephfiguracion@gmail.com](josephfiguracion@gmail.com)')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/josephfiguracion/](https://www.linkedin.com/in/josephfiguracion/)')

    st.subheader('John Barrion - Mentor')
    st.markdown('- Email: barrionjohn@gmail.com')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/johnbarrion/](https://www.linkedin.com/in/johnbarrion/)')