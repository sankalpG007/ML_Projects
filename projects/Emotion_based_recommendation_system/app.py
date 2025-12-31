# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd

from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64

def load_data():
    df = pd.read_csv("muse_v3.csv")
    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']
    df = df[['name','emotional','pleasant','link','artist']]
    df = df.sort_values(by=["emotional", "pleasant"])
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()



df_sad = df[:18000]
df_Fearful = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_Happy = df[72000:]

def fun(list):

    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'Fearful':
            data = pd.concat([data, df_Fearful.sample(n=t)], ignore_index=True)
        elif v == 'Happy':
            data = pd.concat([data, df_Happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':              
                data = pd.concat([data, df_Fearful.sample(n=t)], ignore_index=True)
            elif v == 'Happy':             
                data = pd.concat([data, df_Happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)): 
            v = list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':             
                data = pd.concat([data, df_Fearful.sample(n=t)], ignore_index=True)
            elif v == 'Happy':               
                data = pd.concat([data, df_Happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':              
                data = pd.concat([data, df_Fearful.sample(n=t)], ignore_index=True)
            elif v == 'Happy':               
                data =pd.concat([data, df_Happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):           
            v = list[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':           
                data = pd.concat([data, df_Fearful.sample(n=t)], ignore_index=True)
            elif v == 'Happy':          
                data = pd.concat([data, df_Happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul= list(dict.fromkeys(result))
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul
    

def load_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('model.h5')
    return model

model = load_model()

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}




cv2.ocl.setUseOpenCL(False)


print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>"
            , unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>"
            , unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

list = []
new_df = pd.DataFrame()
with col1:
    pass
with col2:
     img_file = st.camera_input("📷 Take a picture to detect your emotion")

if img_file is not None:
    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected_emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        detected_emotions.append(emotion_dict[max_index])

    if len(detected_emotions) > 0:
        main_emotion = detected_emotions[0]
        st.success(f"✅ Detected Emotion: {main_emotion}")

        detected_list = pre(detected_emotions)
        new_df = fun(detected_list)

        if new_df.empty:
            st.warning("No songs found for this emotion. Try scanning again.")
        else:
            st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>", unsafe_allow_html=True)
            st.write("---------------------------------------------------------------------------------------------------------------------")
        if not new_df.empty:
            for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):
                st.markdown(f"<h4 style='text-align: center;'><a href='{l}'>{i+1}. {n}</a></h4>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
                st.write("---------------------------------------------------------------------------------------------------------------------")
        else:
            st.info("📷 Take a picture or upload an image to detect emotion and get song recommendations.")
    else:
        st.warning("😕 No face detected. Please try again with better lighting or facing the camera.")

    if not new_df.empty and "link" in new_df.columns:
       
        st.write("---------------------------------------------------------------------------------------------------------------------")

    
        for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):
            st.markdown(f"<h4 style='text-align: center;'><a href='{l}'>{i+1}. {n}</a></h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
            st.write("---------------------------------------------------------------------------------------------------------------------")
    else:
        st.info("Take a picture to detect your emotion and get music recommendations 🎵")

        

